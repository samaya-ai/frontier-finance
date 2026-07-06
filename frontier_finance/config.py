from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import ClassVar

import yaml


@dataclass(frozen=True, kw_only=True)
class EvalConfig:
    _REQUIRED_KEYS: ClassVar[tuple[str, ...]] = (
        "rubrics_path",
        "responses_path",
        "output_dir",
    )
    rubrics_path: str
    responses_path: str
    output_dir: str

    # Judge panel. Each rubric is voted on by every model; majority wins, with
    # the first model breaking ties. Models route to a provider by name prefix
    # (claude-* / gpt-*|o* / gemini-*).
    judge_models: list[str] = field(default_factory=lambda: ["claude-opus-4-8"])

    # Rubrics are sent to the judge in batches of this size per LLM call.
    max_rubrics_per_call: int = 30

    # Number of queries graded concurrently (one worker thread per query).
    concurrency: int = 8

    # Max output tokens per judge call. Matches the reference default; the
    # Anthropic judge streams, so this stays under no non-streaming timeout cap.
    max_tokens: int = 64000

    # Per-request timeout (seconds) and SDK-level retry count for the clients.
    # `max_retries` covers transport/API failures; `max_json_parse_retries`
    # covers a successful call whose body can't be parsed as JSON (re-prompted
    # with a format reminder). Set the latter to 0 to disable parse retries.
    request_timeout: float = 60.0
    max_retries: int = 2
    max_json_parse_retries: int = 1

    @classmethod
    def load(cls, path: str) -> EvalConfig:
        """Load an :class:`EvalConfig` from a YAML file.

        Raises ``ValueError`` if the file is not a mapping, a required key is
        missing, or an unknown key is present (so typos fail loudly instead of
        being silently ignored).
        """
        with open(path) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(
                f"Config {path!r} must be a YAML mapping, got {type(raw).__name__}."
            )

        missing = [k for k in cls._REQUIRED_KEYS if k not in raw]
        if missing:
            raise ValueError(f"Config {path!r} is missing required keys: {missing}.")

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"Config {path!r} has unknown keys: {unknown}. Allowed: {sorted(known)}."
            )

        return cls(**raw)
