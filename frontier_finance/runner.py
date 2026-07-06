"""End-to-end run: load inputs, grade concurrently, aggregate, write outputs."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tqdm import tqdm

from frontier_finance.config import EvalConfig
from frontier_finance.data import DataLoader
from frontier_finance.grading import Grader
from frontier_finance.judges import Judge
from frontier_finance.metrics import MetricsReport
from frontier_finance.models import ItemResult

logger = logging.getLogger(__name__)


class EvalRunner:
    """Drives a full evaluation run from an :class:`EvalConfig`."""

    def __init__(self, cfg: EvalConfig) -> None:
        self._cfg = cfg

    def run(self) -> dict[str, Any]:
        """Run the evaluation and return the aggregate metrics dict.

        Side effects: writes ``metrics.json`` and ``per_item.json`` to
        ``cfg.output_dir`` and logs a markdown metrics table.
        """
        cfg = self._cfg
        items = DataLoader(cfg.rubrics_path, cfg.responses_path).load()
        if not items:
            raise ValueError(
                "No queries to evaluate after joining rubrics and responses."
            )

        judges = [
            Judge.build(
                model,
                max_tokens=cfg.max_tokens,
                timeout=cfg.request_timeout,
                max_retries=cfg.max_retries,
            )
            for model in cfg.judge_models
        ]
        grader = Grader(judges, cfg.max_rubrics_per_call, cfg.max_json_parse_retries)
        logger.info("grading %d queries with judges: %s", len(items), cfg.judge_models)

        with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
            # pool.map yields in submission order, so results stay aligned to
            # items; tqdm advances as each ordered result becomes ready.
            results: list[ItemResult] = list(
                tqdm(
                    pool.map(grader.grade, items),
                    total=len(items),
                    desc="grading",
                    unit="query",
                )
            )

        metrics = MetricsReport(results).compute()

        os.makedirs(cfg.output_dir, exist_ok=True)
        self._write_json(os.path.join(cfg.output_dir, "metrics.json"), metrics)
        self._write_json(
            os.path.join(cfg.output_dir, "per_item.json"), self._per_item(results)
        )

        logger.info("evaluation metrics:\n%s", MetricsReport.to_markdown_table(metrics))
        return metrics

    @staticmethod
    def _per_item(results: list[ItemResult]) -> list[dict[str, Any]]:
        out = []
        for r in results:
            # Failed queries have no labels; report each rubric as ungraded (None).
            labels = r.labels if not r.failed else [None] * r.num_rubrics
            out.append(
                {
                    "query_id": r.query_id,
                    "failed": r.failed,
                    "failure_reason": r.failure_reason,
                    "failed_checks_by_judge": r.failed_checks_by_judge,
                    "num_rubrics": r.num_rubrics,
                    "num_qualified": r.num_qualified,
                    "judgements": [
                        {
                            "rubric_id": rubric.rubric_id,
                            "must_have": rubric.must_have,
                            "rubric_type": rubric.rubric_type,
                            "data_source_type": rubric.data_source_type,
                            "qualified": label,
                        }
                        for rubric, label in zip(r.rubrics, labels, strict=True)
                    ],
                }
            )
        return out

    @staticmethod
    def _write_json(path: str, payload: Any) -> None:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
