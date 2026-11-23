"""LLM judge clients.

A :class:`Judge` turns a (system, user) prompt pair into raw response text;
parsing into per-rubric labels is handled by the caller. Models route to a
provider by name prefix via :meth:`Judge.build`. Provider SDKs are imported
lazily.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

# Newer Claude models reject temperature/top_p/top_k with a 400; older ones
# accept temperature=0. Matches the (Opus 4.7+ / Sonnet 5 / Fable / Mythos)
# rule the internal client applies.
_CLAUDE_VERSIONED_RE = re.compile(r"claude-(opus|sonnet)-(\d+)(?:-(\d+))?")

GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Snapshot suffixes a provider may append to the requested model name in the
# response: gpt-5.4-2026-03-05, gpt-4-0613, gemini-2.0-flash-001,
# gemini-2.5-pro-preview-06-05, gemini-2.5-flash-preview-09-2025.
_SNAPSHOT_SUFFIX = r"(?:-\d{4}-\d{2}-\d{2}|-\d{2}-\d{2}(?:\d{2})?|-\d{3,4})?"


class Judge(ABC):
    """Base for a single-model judge client."""

    def __init__(
        self, model: str, *, max_tokens: int, timeout: float, max_retries: int
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response to the (system, user) prompt."""

    @classmethod
    def build(
        cls, model: str, *, max_tokens: int, timeout: float, max_retries: int
    ) -> Judge:
        """Construct the right judge subclass for ``model`` by name prefix."""
        kwargs = {
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if model.startswith("claude-"):
            return AnthropicJudge(model, **kwargs)
        if model.startswith("gpt-") or model.startswith("o"):
            return OpenAIJudge(model, **kwargs)
        if model.startswith("gemini-"):
            return GeminiJudge(model, **kwargs)
        raise ValueError(
            f"Unsupported judge model {model!r}: expected a claude-*, gpt-*/o*, or gemini-* model."
        )

    @staticmethod
    def _require_env(var: str, model: str) -> str:
        value = os.environ.get(var)
        if not value:
            raise RuntimeError(
                f"{var} must be set in the environment to use judge model {model!r}."
            )
        return value

    def _sdk_error(self, extra: str, package: str) -> ImportError:
        """A clear error pointing at the install flavor for a missing backend."""
        return ImportError(
            f"The {package!r} package is required for judge model {self.model!r} but is not installed. "
            f"Install the flavor with: uv pip install 'frontier-finance[{extra}]'."
        )


class AnthropicJudge(Judge):
    """Claude judge via the Anthropic Messages API.

    Judges deterministically: temperature=0 on models that accept it, omitted on
    the newer models (Opus 4.7+, Sonnet 5+, Fable, Mythos) that 400 on sampling
    params. Uses streaming because the SDK refuses non-streaming requests whose
    max_tokens could exceed its ~10-minute estimate (raises ValueError).
    """

    def __init__(
        self, model: str, *, max_tokens: int, timeout: float, max_retries: int
    ) -> None:
        super().__init__(
            model, max_tokens=max_tokens, timeout=timeout, max_retries=max_retries
        )
        try:
            import anthropic
        except ImportError as e:
            raise self._sdk_error("anthropic", "anthropic") from e

        self._require_env("ANTHROPIC_API_KEY", model)
        self._client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)

    @staticmethod
    def _claude_rejects_sampling_params(model: str) -> bool:
        """True for Claude models that 400 on temperature/top_p/top_k.

        Opus 4.7+, Sonnet 5+, Fable, and Mythos reject non-default sampling
        parameters entirely; older Claude models (Opus 4.6, Sonnet 4.x, Haiku)
        accept temperature=0.
        """
        if model.startswith(("claude-fable-", "claude-mythos-")):
            return True
        m = _CLAUDE_VERSIONED_RE.match(model)
        if not m:
            return False
        family, major, minor = m.group(1), int(m.group(2)), int(m.group(3) or 0)
        if family == "opus":
            return (major, minor) >= (4, 7)
        return family == "sonnet" and major >= 5

    def complete(self, system: str, user: str) -> str:
        kwargs: dict = {}
        if not self._claude_rejects_sampling_params(self.model):
            kwargs["temperature"] = 0
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        ) as stream:
            message = stream.get_final_message()
        return "".join(block.text for block in message.content if block.type == "text")


class OpenAIJudge(Judge):
    """GPT / reasoning-model judge via the OpenAI Chat Completions API."""

    def __init__(
        self, model: str, *, max_tokens: int, timeout: float, max_retries: int
    ) -> None:
        super().__init__(
            model, max_tokens=max_tokens, timeout=timeout, max_retries=max_retries
        )
        try:
            import openai
        except ImportError as e:
            raise self._sdk_error("openai", "openai") from e

        self._client = openai.OpenAI(
            timeout=timeout, max_retries=max_retries, **self._client_kwargs()
        )

    def _client_kwargs(self) -> dict:
        """Endpoint/credential args for the client; the SDK reads OPENAI_* env."""
        self._require_env("OPENAI_API_KEY", self.model)
        return {}

    def _request_kwargs(self) -> dict:
        # Pin temperature=0 for deterministic judging on gpt-* chat models. The
        # o* reasoning models 400 on any temperature != 1, so it's omitted there.
        if self.model.startswith("gpt-"):
            return {"temperature": 0}
        return {}

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=self._max_tokens,
            **self._request_kwargs(),
        )
        self._check_served_model(response.model)
        return response.choices[0].message.content or ""

    def _check_served_model(self, served: str | None) -> None:
        """Fail if a different model answered (e.g. a gateway fallback).

        Accepts the requested name, optionally with a ``models/`` or
        ``<provider>/`` prefix and a dated/numbered snapshot suffix.
        """
        name = (served or "").rsplit("/", 1)[-1]
        if not re.fullmatch(rf"{re.escape(self.model)}{_SNAPSHOT_SUFFIX}", name):
            raise RuntimeError(
                f"Judge {self.model!r} was answered by model {served!r}; "
                "refusing a substitute model's verdict."
            )


class GeminiJudge(OpenAIJudge):
    """Gemini judge via the OpenAI-compatible Chat Completions API.

    Defaults to Google's OpenAI-compatible endpoint with a Gemini API key. Set
    GEMINI_BASE_URL (and a matching GEMINI_API_KEY) to go through any
    OpenAI-compatible gateway instead. Uses its own env vars, not OPENAI_*, so
    GPT and Gemini judges can point at different endpoints.
    """

    def _client_kwargs(self) -> dict:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"GEMINI_API_KEY (or GOOGLE_API_KEY) must be set to use judge model {self.model!r}."
            )
        return {
            "api_key": api_key,
            "base_url": os.environ.get("GEMINI_BASE_URL") or GEMINI_DEFAULT_BASE_URL,
        }

    def _request_kwargs(self) -> dict:
        kwargs: dict = {"temperature": 0}
        # gemini-3 supports thinking-level control; pro models require at least
        # low, others accept minimal. Keep thinking shallow so it doesn't eat the
        # output budget on what is a mechanical labelling task.
        if self.model.startswith("gemini-3"):
            kwargs["reasoning_effort"] = "low" if "pro" in self.model else "minimal"
        return kwargs
