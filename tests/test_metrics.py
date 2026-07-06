from frontier_finance.metrics import MetricsReport
from frontier_finance.models import ItemResult, Rubric


def _rubric(rid, must_have, rtype="t", dst="s"):
    return Rubric(
        rubric_id=rid,
        rubric_text=f"r{rid}",
        must_have=must_have,
        rubric_type=rtype,
        data_source_type=dst,
    )


def test_aggregate_basic_rates():
    # Query A: 2 rubrics, 1 qualified (must_have qualified). Query B: 2 rubrics, 2 qualified.
    a = ItemResult(
        query_id="A",
        rubrics=[_rubric(1, True), _rubric(2, False)],
        labels=[True, False],
    )
    b = ItemResult(
        query_id="B",
        rubrics=[_rubric(1, True), _rubric(2, False)],
        labels=[True, True],
    )
    m = MetricsReport([a, b]).compute()

    assert m["num_records"] == 2
    assert m["num_failed_queries"] == 0
    assert m["num_rubrics_on_success_queries"] == 4
    assert m["num_qualified_rubrics"] == 3

    # micro = 3/4 ; macro = mean(0.5, 1.0) = 0.75
    assert m["micro_avg_qualification_rate_on_success_queries"] == 0.75
    assert m["macro_avg_qualification_rate_on_success_queries"] == 0.75

    # must_have: A has 1 must_have qualified, B has 1 must_have qualified -> 2/2
    assert m["micro_avg_qualification_rate_must_have_on_success_queries"] == 1.0
    assert m["macro_avg_qualification_rate_must_have_on_success_queries"] == 1.0


def test_failed_query_affects_all_but_not_success():
    ok = ItemResult(query_id="A", rubrics=[_rubric(1, True)], labels=[True])
    bad = ItemResult(
        query_id="B", rubrics=[_rubric(1, True), _rubric(2, False)], failed=True
    )
    m = MetricsReport([ok, bad]).compute()

    assert m["num_failed_queries"] == 1
    assert m["num_rubrics_on_success_queries"] == 1
    assert m["num_rubrics_on_all_queries"] == 3
    assert m["num_qualified_rubrics"] == 1

    # success: 1/1 = 1.0
    assert m["micro_avg_qualification_rate_on_success_queries"] == 1.0
    # all queries micro: 1 qualified / 3 rubrics
    assert m["micro_avg_qualification_rate_on_all_queries"] == 1 / 3
    # all queries macro: mean(1.0, 0.0) over 2 records = 0.5
    assert m["macro_avg_qualification_rate_on_all_queries"] == 0.5


def test_failed_checks_by_judge_and_error_counts():
    ok = ItemResult(
        query_id="A",
        rubrics=[_rubric(1, True)],
        labels=[True],
        failed_checks_by_judge={"j1": 3},  # j1 flaked on this graded query too
    )
    judge_err = ItemResult(
        query_id="B",
        rubrics=[_rubric(1, True), _rubric(2, False)],
        failed=True,
        failure_reason="judge_error",
        failed_checks_by_judge={"j1": 2, "j2": 2},
    )
    no_resp = ItemResult(
        query_id="C",
        rubrics=[_rubric(1, True)],
        failed=True,
        failure_reason="no_response",
    )
    m = MetricsReport([ok, judge_err, no_resp]).compute()

    # judge_error (B) is excluded from the scored set: only A and C count.
    assert m["num_records"] == 2
    assert m["num_judge_errors"] == 1
    assert m["num_failed_queries"] == 1  # only the no_response query (C)
    # ...but its skipped checks still show up in the per-judge stats.
    assert m["failed_criteria_checks_by_judge"] == {"j1": 5, "j2": 2}


def test_breakdowns():
    r = ItemResult(
        query_id="A",
        rubrics=[
            _rubric(1, True, rtype="fact", dst="web"),
            _rubric(2, False, rtype="fact", dst="filing"),
            _rubric(3, False, rtype="opinion", dst="web"),
        ],
        labels=[True, False, True],
    )
    m = MetricsReport([r]).compute()
    by_type = m["breakdown_by_rubric_type"]
    assert by_type["fact"] == {
        "num_rubrics": 2,
        "num_qualified": 1,
        "qualification_rate": 0.5,
    }
    assert by_type["opinion"] == {
        "num_rubrics": 1,
        "num_qualified": 1,
        "qualification_rate": 1.0,
    }

    by_src = m["breakdown_by_data_source_type"]
    assert by_src["web"]["num_qualified"] == 2
    assert by_src["filing"]["num_qualified"] == 0
