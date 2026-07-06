import json
import re

import pytest

from frontier_finance.grading import Grader
from frontier_finance.models import EvalItem, Rubric


class _FakeJudge:
    """Returns a fixed label for every rubric index in the prompt."""

    def __init__(self, model, label):
        self.model = model
        self._label = label

    def complete(self, system, user):
        count = len(re.findall(r"(?m)^\d+\. ", user))
        return json.dumps({str(i): {"label": self._label} for i in range(count)})


class _FailingJudge:
    """A judge whose call always errors, so it's dropped from the vote."""

    def __init__(self, model):
        self.model = model

    def complete(self, system, user):
        raise RuntimeError("judge unavailable")


class _ScriptedJudge:
    """Returns a queued response per call, recording the prompts it received."""

    def __init__(self, model, responses):
        self.model = model
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, system, user):
        self.prompts.append(user)
        return self._responses.pop(0) if self._responses else "still garbage"

    @staticmethod
    def labels_json(user, label):
        count = len(re.findall(r"(?m)^\d+\. ", user))
        return json.dumps({str(i): {"label": label} for i in range(count)})


def _item(response="resp"):
    return EvalItem(
        query_id="q1",
        query="Q",
        query_date="2024-01-01",
        rubrics=[
            Rubric(rubric_id=1, rubric_text="a", must_have=True),
            Rubric(rubric_id=2, rubric_text="b", must_have=False),
        ],
        system_response=response,
    )


def test_grade_majority_vote():
    judges = [_FakeJudge("m1", True), _FakeJudge("m2", True), _FakeJudge("m3", False)]
    result = Grader(judges, max_rubrics_per_call=30).grade(_item())
    assert result.failed is False
    assert result.labels == [True, True]  # 2/3 say True


def test_grade_no_response_is_failed():
    item = _item(response=None)
    result = Grader([_FakeJudge("m1", True)], max_rubrics_per_call=30).grade(item)
    assert result.failed is True
    assert result.failure_reason == "no_response"
    assert result.labels == []
    assert result.failed_checks_by_judge == {}


def test_grade_all_judges_fail_is_judge_error():
    # Response exists but every judge errors -> judge_error, and each judge's
    # skipped checks (2 rubrics on _item) are recorded.
    judges = [_FailingJudge("a"), _FailingJudge("b")]
    result = Grader(judges, max_rubrics_per_call=30, max_json_parse_retries=0).grade(
        _item()
    )
    assert result.failed is True
    assert result.failure_reason == "judge_error"
    assert result.labels == []
    assert result.failed_checks_by_judge == {"a": 2, "b": 2}


def test_grade_partial_judge_failure_grades_on_survivors():
    # One of three judges fails; the query is still graded on the surviving two,
    # and the failed judge's skipped checks are recorded.
    judges = [_FailingJudge("bad"), _FakeJudge("m2", True), _FakeJudge("m3", True)]
    result = Grader(judges, max_rubrics_per_call=30, max_json_parse_retries=0).grade(
        _item()
    )
    assert result.failed is False
    assert result.labels == [True, True]
    assert result.failed_checks_by_judge == {"bad": 2}


def test_grader_requires_a_judge():
    with pytest.raises(ValueError):
        Grader([], max_rubrics_per_call=30)


def test_grader_rejects_negative_parse_retries():
    with pytest.raises(ValueError):
        Grader(
            [_FakeJudge("m1", True)], max_rubrics_per_call=30, max_json_parse_retries=-1
        )


def test_unparseable_judge_recovers_after_retry():
    valid = '{"0": {"label": true}, "1": {"label": true}}'
    judge = _ScriptedJudge("m1", ["not json at all", valid])
    result = Grader([judge], max_rubrics_per_call=30, max_json_parse_retries=1).grade(
        _item()
    )

    assert result.failed is False
    assert result.labels == [True, True]
    # Re-prompted exactly once, and the format reminder appears only on the retry.
    assert len(judge.prompts) == 2
    from frontier_finance.prompts import RubricPrompt

    assert RubricPrompt.JSON_FORMAT_RETRY_SUFFIX not in judge.prompts[0]
    assert RubricPrompt.JSON_FORMAT_RETRY_SUFFIX in judge.prompts[1]


def test_unparseable_judge_dropped_after_exhausting_retries():
    # Sole judge never returns valid JSON -> dropped from the vote -> item failed,
    # NOT silently scored as all-False.
    judge = _ScriptedJudge("m1", ["garbage", "more garbage"])
    result = Grader([judge], max_rubrics_per_call=30, max_json_parse_retries=1).grade(
        _item()
    )

    assert result.failed is True
    assert result.labels == []
    assert len(judge.prompts) == 2  # initial + 1 retry


def test_parse_retries_disabled_makes_a_single_call():
    judge = _ScriptedJudge("m1", ["garbage"])
    result = Grader([judge], max_rubrics_per_call=30, max_json_parse_retries=0).grade(
        _item()
    )

    assert result.failed is True
    assert len(judge.prompts) == 1  # no retry


def test_unparseable_judge_dropped_but_others_still_vote():
    bad = _ScriptedJudge("m1", ["garbage", "garbage"])
    result = Grader(
        [bad, _FakeJudge("m2", True), _FakeJudge("m3", True)],
        max_rubrics_per_call=30,
        max_json_parse_retries=1,
    ).grade(_item())

    # m1 is dropped; m2 and m3 carry the vote rather than m1 dragging it to False.
    assert result.failed is False
    assert result.labels == [True, True]


def test_majority_vote_unanimous():
    assert Grader.majority_vote([[True, False], [True, False]]) == [True, False]


def test_majority_vote_wins():
    assert Grader.majority_vote([[True], [True], [False]]) == [True]


def test_majority_vote_tie_defers_to_first():
    # Even-panel tie is broken by the first model's label.
    assert Grader.majority_vote([[True], [False]]) == [True]
    assert Grader.majority_vote([[False], [True]]) == [False]


def test_majority_vote_single_model():
    assert Grader.majority_vote([[True, True, False]]) == [True, True, False]


def test_majority_vote_mismatched_widths_raise():
    with pytest.raises(ValueError):
        Grader.majority_vote([[True, False], [True]])


def test_majority_vote_empty_raises():
    with pytest.raises(ValueError):
        Grader.majority_vote([])
