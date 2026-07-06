import json

from frontier_finance.data import DataLoader


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_join_marks_missing_response_failed(tmp_path):
    rubrics_path = tmp_path / "rubrics.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    _write_jsonl(
        rubrics_path,
        [
            {
                "query_id": "q1",
                "query": "Q1",
                "query_date": "2024-01-01",
                "rubrics": [{"rubric_id": 1, "rubric_text": "r", "must_have": True}],
            },
            {
                "query_id": "q2",
                "query": "Q2",
                "query_date": "2024-01-01",
                "rubrics": [{"rubric_id": 1, "rubric_text": "r", "must_have": False}],
            },
        ],
    )
    # q2 has no response; an orphan response q3 is ignored.
    responses_path.write_text(
        json.dumps(
            [
                {"query_id": "q1", "system_summary": "answer"},
                {"query_id": "q3", "system_summary": "orphan"},
            ]
        )
    )

    items = DataLoader(str(rubrics_path), str(responses_path)).load()
    by_id = {it.query_id: it for it in items}
    assert set(by_id) == {"q1", "q2"}
    assert by_id["q1"].system_response == "answer"
    assert by_id["q2"].system_response is None


def test_invalid_json_line_raises(tmp_path):
    import pytest

    rubrics_path = tmp_path / "rubrics.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    rubrics_path.write_text("{not valid json}\n")
    responses_path.write_text("")
    with pytest.raises(ValueError):
        DataLoader(str(rubrics_path), str(responses_path)).load()
