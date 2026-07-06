import pytest

from frontier_finance.judges import AnthropicJudge, Judge

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
