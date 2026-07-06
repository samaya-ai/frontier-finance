import pytest

from frontier_finance.judges import Judge

_BUILD_KWARGS = {"max_tokens": 100, "timeout": 5.0, "max_retries": 0}


def test_build_unknown_model_raises():
    with pytest.raises(ValueError):
        Judge.build("llama-3", **_BUILD_KWARGS)


def test_build_routes_to_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    judge = Judge.build("claude-opus-4-8", **_BUILD_KWARGS)
    assert type(judge).__name__ == "AnthropicJudge"
    assert judge.model == "claude-opus-4-8"


def test_anthropic_judge_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        Judge.build("claude-opus-4-8", **_BUILD_KWARGS)
