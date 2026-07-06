from frontier_finance.prompts import RubricPrompt


def test_try_parse_fenced_json():
    text = 'here you go:\n```json\n{"0": {"reason": "yes", "label": true}}\n```'
    assert RubricPrompt.try_parse(text) == {"0": {"reason": "yes", "label": True}}


def test_try_parse_raw_json_with_prose():
    text = 'Sure. {"0": {"label": true}, "1": {"label": false}} done.'
    parsed = RubricPrompt.try_parse(text)
    assert parsed["0"]["label"] is True
    assert parsed["1"]["label"] is False


def test_try_parse_returns_none_on_failure():
    assert RubricPrompt.try_parse("no json at all") is None
    assert RubricPrompt.try_parse("") is None


def test_try_parse_distinguishes_empty_object_from_failure():
    # A model that legitimately judged nothing as qualifying parses to {}, which
    # must be distinguishable from an unparseable reply (None).
    assert RubricPrompt.try_parse("{}") == {}
    assert RubricPrompt.try_parse("```json\n{}\n```") == {}


def test_build_user_numbers_rubrics_zero_indexed():
    from frontier_finance.models import Rubric

    rubrics = [
        Rubric(rubric_id=7, rubric_text="alpha", must_have=True),
        Rubric(rubric_id=9, rubric_text="beta", must_have=False),
    ]
    full = RubricPrompt.build_user("Q", "2024-01-01", "resp", rubrics)
    assert "0. alpha" in full
    assert "1. beta" in full
    assert "<report>\nresp\n</report>" in full


def test_build_user_embeds_report_exactly_once():
    from frontier_finance.models import Rubric

    rubrics = [Rubric(rubric_id=1, rubric_text="alpha", must_have=True)]
    full = RubricPrompt.build_user("Q", "2024-01-01", "the-report", rubrics)
    assert full.count("<report>\nthe-report\n</report>") == 1
