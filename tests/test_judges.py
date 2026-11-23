import pytest

from frontier_finance.judges import (
    GEMINI_DEFAULT_BASE_URL,
    AnthropicJudge,
    GeminiJudge,
    Judge,
)

_BUILD_KWARGS = {"max_tokens": 100, "timeout": 5.0, "max_retries": 0}


def test_build_unknown_model_raises():
    with pytest.raises(ValueError):
        Judge.build("llama-3", **_BUILD_KWARGS)


@pytest.mark.parametrize(
    "model, rejects",
    [
        ("claude-opus-4-8", True),  # opus >= 4.7 rejects sampling params
        ("claude-opus-4-7", True),
        ("claude-opus-4-6", False),  # older opus accepts temperature=0
        ("claude-sonnet-5", True),  # sonnet >= 5 rejects
        ("claude-sonnet-4-6", False),
        ("claude-fable-1", True),
        ("claude-mythos-1", True),
        ("claude-haiku-4-5", False),
        ("claude-3-5-sonnet", False),  # unversioned-by-this-scheme -> accepts
    ],
)
def test_claude_rejects_sampling_params(model, rejects):
    assert AnthropicJudge._claude_rejects_sampling_params(model) is rejects


def test_build_routes_to_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    judge = Judge.build("claude-opus-4-8", **_BUILD_KWARGS)
    assert type(judge).__name__ == "AnthropicJudge"
    assert judge.model == "claude-opus-4-8"


def test_anthropic_judge_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        Judge.build("claude-opus-4-8", **_BUILD_KWARGS)


class _FakeOpenAI:
    """Stands in for openai.OpenAI; records constructor and create() kwargs."""

    instances: list["_FakeOpenAI"] = []
    # Model name the fake server reports; None echoes the requested model.
    served_model: str | None = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.create_kwargs = None
        self.chat = self
        self.completions = self
        _FakeOpenAI.instances.append(self)

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        message = type("Message", (), {"content": '{"0": {"label": true}}'})()
        choice = type("Choice", (), {"message": message})()
        served = self.served_model or kwargs["model"]
        return type("Response", (), {"choices": [choice], "model": served})()


@pytest.fixture
def fake_openai(monkeypatch):
    import openai

    _FakeOpenAI.instances = []
    _FakeOpenAI.served_model = None
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    for var in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    return _FakeOpenAI


def _complete(judge):
    assert judge.complete("sys", "usr") == '{"0": {"label": true}}'
    return judge._client.create_kwargs


def test_build_routes_to_gemini(fake_openai, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    judge = Judge.build("gemini-3.1-pro-preview", **_BUILD_KWARGS)
    assert isinstance(judge, GeminiJudge)
    assert judge.model == "gemini-3.1-pro-preview"


def test_gemini_judge_defaults_to_google_openai_endpoint(fake_openai, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    judge = Judge.build("gemini-3-flash-preview", **_BUILD_KWARGS)
    assert judge._client.init_kwargs == {
        "api_key": "gem-key",
        "base_url": GEMINI_DEFAULT_BASE_URL,
        "timeout": 5.0,
        "max_retries": 0,
    }
    assert GEMINI_DEFAULT_BASE_URL == (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )


def test_gemini_judge_base_url_override(fake_openai, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gateway-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://gateway.example/v1")
    judge = Judge.build("gemini-3-flash-preview", **_BUILD_KWARGS)
    assert judge._client.init_kwargs["base_url"] == "https://gateway.example/v1"
    assert judge._client.init_kwargs["api_key"] == "gateway-key"


def test_gemini_judge_google_api_key_fallback(fake_openai, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    judge = Judge.build("gemini-3-flash-preview", **_BUILD_KWARGS)
    assert judge._client.init_kwargs["api_key"] == "google-key"


def test_gemini_judge_prefers_gemini_api_key(fake_openai, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    judge = Judge.build("gemini-3-flash-preview", **_BUILD_KWARGS)
    assert judge._client.init_kwargs["api_key"] == "gem-key"


def test_gemini_judge_requires_key(fake_openai, monkeypatch):
    # An OpenAI key must not satisfy the Gemini judge.
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        Judge.build("gemini-3-flash-preview", **_BUILD_KWARGS)


def test_gemini_judge_ignores_openai_env(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-proxy.example/v1")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    judge = Judge.build("gemini-3-flash-preview", **_BUILD_KWARGS)
    assert judge._client.init_kwargs["api_key"] == "gem-key"
    assert judge._client.init_kwargs["base_url"] == GEMINI_DEFAULT_BASE_URL


def test_openai_judge_ignores_gemini_env(fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://gateway.example/v1")
    judge = Judge.build("gpt-5.4", **_BUILD_KWARGS)
    # No explicit endpoint/key: the SDK resolves OPENAI_API_KEY / OPENAI_BASE_URL.
    assert judge._client.init_kwargs == {"timeout": 5.0, "max_retries": 0}


@pytest.mark.parametrize(
    "model, effort",
    [
        ("gemini-3.1-pro-preview", "low"),  # pro needs at least low
        ("gemini-3-pro-preview", "low"),
        ("gemini-3-flash-preview", "minimal"),
        ("gemini-3.1-flash-lite", "minimal"),
        ("gemini-2.5-pro", None),  # pre-3: no thinking control, model default
        ("gemini-2.5-flash", None),
    ],
)
def test_gemini_judge_request(fake_openai, monkeypatch, model, effort):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    kwargs = _complete(Judge.build(model, **_BUILD_KWARGS))
    expected = {
        "model": model,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ],
        "max_completion_tokens": 100,
        "temperature": 0,
    }
    if effort is not None:
        expected["reasoning_effort"] = effort
    assert kwargs == expected


@pytest.mark.parametrize(
    "model, extra",
    [("gpt-5.4", {"temperature": 0}), ("o3", {})],
)
def test_openai_judge_request(fake_openai, monkeypatch, model, extra):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    kwargs = _complete(Judge.build(model, **_BUILD_KWARGS))
    assert kwargs == {
        "model": model,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ],
        "max_completion_tokens": 100,
        **extra,
    }


def test_openai_judge_requires_key(fake_openai):
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Judge.build("gpt-5.4", **_BUILD_KWARGS)


@pytest.mark.parametrize(
    "requested, served",
    [
        ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview"),
        ("gemini-3.1-pro-preview", "models/gemini-3.1-pro-preview"),
        ("gemini-2.0-flash", "gemini-2.0-flash-001"),
        ("gpt-5.4", "gpt-5.4-2026-03-05"),
        ("gpt-4", "gpt-4-0613"),
    ],
)
def test_served_model_accepts_same_model(fake_openai, monkeypatch, requested, served):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    fake_openai.served_model = served
    _complete(Judge.build(requested, **_BUILD_KWARGS))


@pytest.mark.parametrize(
    "requested, served",
    [
        # Gateway fallbacks answering with a different model.
        ("gemini-3.1-pro-preview", "gemini-2.5-pro"),
        ("gemini-3.1-pro-preview", "gemini-3.8-flash"),
        ("gemini-3-pro-preview", "gemini-3.1-pro-preview"),
        ("gemini-3-flash-preview", "gemini-3-flash-preview-lite"),
        ("gpt-5.4", "gpt-5.4-mini"),
        ("gpt-5.4", "gpt-4o"),
    ],
)
def test_served_model_rejects_substitute(fake_openai, monkeypatch, requested, served):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    fake_openai.served_model = served
    judge = Judge.build(requested, **_BUILD_KWARGS)
    with pytest.raises(RuntimeError, match="answered by model"):
        judge.complete("sys", "usr")


@pytest.mark.parametrize("served", [None, ""])
def test_served_model_rejects_missing(fake_openai, monkeypatch, served):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    judge = Judge.build("gemini-3-flash-preview", **_BUILD_KWARGS)
    with pytest.raises(RuntimeError, match="answered by model"):
        judge._check_served_model(served)
