"""Aggregate per-query grading results into qualification-rate metrics.

Terminology (matches the source criteria-eval task):
- *qualified*: a rubric the response satisfied (judge label True).
- *macro* average: mean of per-query qualification rates.
- *micro* average: total qualified rubrics / total rubrics.
- *success* query: one that was graded (had a response and at least one judge
  model succeeded). *failed* queries (no response) contribute 0 qualified and
  their full rubric count toward the "all queries" denominators.
- *judge error*: every judge model failed on a present response. This is a
  harness-side failure, not the system's, so it is *excluded* from every scored
  metric (num_records and all denominators) — equivalent to skipping it — and
  only surfaced as a count and in the per-judge failed-check stats.
- *must_have*: the essential subset of rubrics. must_have averages are taken
  only over queries that have at least one must_have rubric — a query with none
  neither qualifies nor fails a must_have and is left out of the denominator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from frontier_finance.models import ItemResult


class MetricsReport:
    """Computes the aggregate metrics for a set of graded results."""

    def __init__(self, results: list[ItemResult]) -> None:
        self._results = results

    def compute(self) -> dict[str, Any]:
        """Compute the full metrics dict from the graded results."""
        # Judge errors are excluded from every scored metric (they're the
        # harness's fault, not the system's); `results` is the scored set.
        judge_errors = [r for r in self._results if r.failure_reason == "judge_error"]
        results = [r for r in self._results if r.failure_reason != "judge_error"]
        success = [r for r in results if not r.failed]
        failed = [r for r in results if r.failed]

        # Totals over successfully-graded queries.
        total_rubrics_success = sum(r.num_rubrics for r in success)
        total_qualified = sum(r.num_qualified for r in success)
        total_rubrics_all = sum(r.num_rubrics for r in results)

        # Essential (must_have) totals.
        mh_total_success = 0
        mh_qualified_success = 0
        for r in success:
            for rubric, label in zip(r.rubrics, r.labels, strict=True):
                if rubric.must_have:
                    mh_total_success += 1
                    mh_qualified_success += int(label)
        mh_total_all = sum(1 for r in results for rb in r.rubrics if rb.must_have)

        # Per-query rates (success only) for macro averages.
        rates = [self._safe_div(r.num_qualified, r.num_rubrics) for r in success]
        mh_rates = [
            self._safe_div(
                sum(
                    int(lbl)
                    for rb, lbl in zip(r.rubrics, r.labels, strict=True)
                    if rb.must_have
                ),
                sum(1 for rb in r.rubrics if rb.must_have),
            )
            for r in success
            if any(rb.must_have for rb in r.rubrics)
        ]

        n_records = len(results)
        mh_n_records = sum(1 for r in results if any(rb.must_have for rb in r.rubrics))

        # Per-judge count of rubric checks the judge failed to produce, summed
        # across ALL queries (including excluded judge errors and partial
        # failures on graded queries) — surfaces a flaky/degraded judge.
        failed_checks_by_judge: dict[str, int] = defaultdict(int)
        for r in self._results:
            for model, count in r.failed_checks_by_judge.items():
                failed_checks_by_judge[model] += count

        return {
            "num_records": n_records,
            "num_failed_queries": len(failed),
            "num_judge_errors": len(judge_errors),
            "failed_criteria_checks_by_judge": dict(
                sorted(failed_checks_by_judge.items())
            ),
            "num_rubrics_on_success_queries": total_rubrics_success,
            "num_rubrics_on_all_queries": total_rubrics_all,
            "num_qualified_rubrics": total_qualified,
            # All rubrics, success queries.
            "macro_avg_qualification_rate_on_success_queries": self._safe_div(
                sum(rates), len(success)
            ),
            "micro_avg_qualification_rate_on_success_queries": self._safe_div(
                total_qualified, total_rubrics_success
            ),
            # All rubrics, all queries (failed contribute 0 numerator, full denominator).
            "macro_avg_qualification_rate_on_all_queries": self._safe_div(
                sum(rates), n_records
            ),
            "micro_avg_qualification_rate_on_all_queries": self._safe_div(
                total_qualified, total_rubrics_all
            ),
            # must_have rubrics, success queries.
            "macro_avg_qualification_rate_must_have_on_success_queries": self._safe_div(
                sum(mh_rates), len(mh_rates)
            ),
            "micro_avg_qualification_rate_must_have_on_success_queries": self._safe_div(
                mh_qualified_success, mh_total_success
            ),
            # must_have rubrics, all queries.
            "macro_avg_qualification_rate_must_have_on_all_queries": self._safe_div(
                sum(mh_rates), mh_n_records
            ),
            "micro_avg_qualification_rate_must_have_on_all_queries": self._safe_div(
                mh_qualified_success, mh_total_all
            ),
            "breakdown_by_rubric_type": self._breakdown(success, key="rubric_type"),
            "breakdown_by_data_source_type": self._breakdown(
                success, key="data_source_type"
            ),
        }

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    @classmethod
    def _breakdown(
        cls, success: list[ItemResult], *, key: str
    ) -> dict[str, dict[str, Any]]:
        """Micro qualification rate per taxonomy value, over success queries."""
        totals: dict[str, int] = defaultdict(int)
        qualified: dict[str, int] = defaultdict(int)
        for r in success:
            for rubric, label in zip(r.rubrics, r.labels, strict=True):
                value = getattr(rubric, key) or "(unspecified)"
                totals[value] += 1
                qualified[value] += int(label)
        return {
            value: {
                "num_rubrics": totals[value],
                "num_qualified": qualified[value],
                "qualification_rate": cls._safe_div(qualified[value], totals[value]),
            }
            for value in sorted(totals)
        }

    @staticmethod
    def to_markdown_table(metrics: dict[str, Any]) -> str:
        """Render the scalar (non-nested) metrics as a two-column markdown table."""
        rows = [(k, v) for k, v in metrics.items() if not isinstance(v, (dict, list))]
        width = max((len(k) for k, _ in rows), default=6)
        lines = [f"| {'metric'.ljust(width)} | value |", f"| {'-' * width} | ----- |"]
        for k, v in rows:
            formatted = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"| {k.ljust(width)} | {formatted} |")
        return "\n".join(lines)
