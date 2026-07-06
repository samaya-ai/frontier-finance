"""Load rubrics and responses and join them on ``query_id``.

Rubrics JSONL — one query per line:
    {"query_id", "query", "query_date", "rubrics": [{"rubric_id", "rubric_text",
     "must_have", "rubric_type", "data_source_type", ...}], ...}

Responses — a JSON array (the criteria_eval ``system_summaries.json``):
    [{"query_id", "system_summary"}, ...]
"""

from __future__ import annotations

import datetime
import json
import logging
from collections.abc import Iterator
from typing import Any

from frontier_finance.models import EvalItem, Rubric

logger = logging.getLogger(__name__)


class DataLoader:
    """Reads the rubrics and responses files and joins them into eval items."""

    def __init__(self, rubrics_path: str, responses_path: str) -> None:
        self._rubrics_path = rubrics_path
        self._responses_path = responses_path

    def load(self) -> list[EvalItem]:
        """Load and join both files into a list of :class:`EvalItem`."""
        return self._join(self._load_rubrics(), self._load_responses())

    @staticmethod
    def _iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
        """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
        with open(path) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e

    def _load_rubrics(self) -> dict[str, EvalItem]:
        """Load the rubrics file into ``{query_id: EvalItem}`` (response unset)."""
        items: dict[str, EvalItem] = {}
        for obj in self._iter_jsonl(self._rubrics_path):
            for key in ("query_id", "query", "query_date"):
                if key not in obj:
                    raise ValueError(
                        f"rubrics entry missing required key {key!r}: {obj}"
                    )
            query_id = str(obj["query_id"])
            rubrics = [Rubric.from_dict(r) for r in obj.get("rubrics", [])]
            if not rubrics:
                logger.warning("query_id %s has no rubrics; skipping", query_id)
                continue
            if query_id in items:
                logger.warning(
                    "duplicate query_id %s in rubrics; keeping the first", query_id
                )
                continue
            query_date = str(obj["query_date"])
            try:
                datetime.datetime.strptime(query_date, "%Y-%m-%d")
            except ValueError as e:
                raise ValueError(
                    f"query_id {query_id}: query_date {query_date!r} is not in YYYY-MM-DD format"
                ) from e
            items[query_id] = EvalItem(
                query_id=query_id,
                query=obj["query"],
                query_date=query_date,
                rubrics=rubrics,
            )
        return items

    def _load_responses(self) -> dict[str, str]:
        """Load the ``system_summaries.json`` array into ``{query_id: summary}``."""
        with open(self._responses_path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"{self._responses_path}: expected a JSON array of responses, "
                f"got {type(data).__name__}."
            )
        responses: dict[str, str] = {}
        for obj in data:
            query_id = str(obj["query_id"])
            if query_id in responses:
                logger.warning(
                    "duplicate query_id %s in responses; keeping the first", query_id
                )
                continue
            responses[query_id] = obj["system_summary"]
        return responses

    @staticmethod
    def _join(
        rubrics_by_id: dict[str, EvalItem], responses_by_id: dict[str, str]
    ) -> list[EvalItem]:
        """Attach responses to rubric items.

        Every rubric query becomes an :class:`EvalItem`; one missing a response
        is kept with ``system_response=None`` (graded as failed). Responses
        without a matching rubric query are dropped with a warning.
        """
        orphan_responses = sorted(set(responses_by_id) - set(rubrics_by_id))
        if orphan_responses:
            logger.warning(
                "%d response(s) have no matching rubric query and were ignored: %s",
                len(orphan_responses),
                orphan_responses[:10],
            )

        items: list[EvalItem] = []
        missing_responses = 0
        for query_id, item in rubrics_by_id.items():
            item.system_response = responses_by_id.get(query_id)
            if item.system_response is None:
                missing_responses += 1
            items.append(item)

        if missing_responses:
            logger.warning(
                "%d rubric query/queries have no response and will be graded as failed",
                missing_responses,
            )
        return items
