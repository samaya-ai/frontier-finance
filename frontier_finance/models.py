"""Core data structures: rubrics, joined eval items, and graded results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class Rubric:
    """A single gradeable criterion attached to a query."""

    rubric_id: int
    rubric_text: str
    must_have: bool
    rubric_type: str = ""
    data_source_type: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Rubric:
        return cls(
            rubric_id=d["rubric_id"],
            rubric_text=d["rubric_text"],
            must_have=bool(d.get("must_have", False)),
            rubric_type=d.get("rubric_type", "") or "",
            data_source_type=d.get("data_source_type", "") or "",
        )


@dataclass(kw_only=True)
class EvalItem:
    """A query with its rubrics and (once joined) the response to grade."""

    query_id: str
    query: str
    query_date: str
    rubrics: list[Rubric]
    system_response: str | None = None


@dataclass(kw_only=True)
class ItemResult:
    """Per-query grading outcome.

    ``labels`` is aligned 1:1 with ``rubrics`` — ``labels[i]`` is whether the
    response satisfied ``rubrics[i]``. ``failed`` is True when the query could
    not be graded; ``failure_reason`` says why:

    - ``"no_response"`` — no ``system_response`` was supplied (a system failure).
    - ``"judge_error"`` — a response existed but *every* judge model errored (a
      harness-side failure, not the system's).

    A partial judge failure (some judges error, at least one succeeds) is NOT a
    failure — the query is graded on the surviving votes. ``failed_checks_by_judge``
    maps a judge model to how many rubric checks it failed to produce for this
    query (the query's full rubric count when that judge errored, else absent).
    """

    query_id: str
    rubrics: list[Rubric]
    labels: list[bool] = field(default_factory=list)
    failed: bool = False
    failure_reason: str | None = None
    failed_checks_by_judge: dict[str, int] = field(default_factory=dict)

    @property
    def num_rubrics(self) -> int:
        return len(self.rubrics)

    @property
    def num_qualified(self) -> int:
        return sum(self.labels)
