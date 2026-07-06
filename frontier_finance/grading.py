from __future__ import annotations

import logging

from frontier_finance.judges import Judge
from frontier_finance.models import EvalItem, ItemResult, Rubric
from frontier_finance.prompts import RubricPrompt

logger = logging.getLogger(__name__)


class Grader:
    def __init__(
        self,
        judges: list[Judge],
        max_rubrics_per_call: int,
        max_json_parse_retries: int = 1,
    ) -> None:
        if not judges:
            raise ValueError("Grader needs at least one judge.")
        if max_json_parse_retries < 0:
            raise ValueError("max_json_parse_retries must be non-negative.")
        self._judges = judges
        self._max_per_call = max_rubrics_per_call
        self._max_json_parse_retries = max_json_parse_retries

    def grade(self, item: EvalItem) -> ItemResult:
        """Grade a single query against its rubrics using the judge panel.

        A missing response fails as ``no_response``. Judges that error are
        dropped and their skipped rubric checks recorded per judge; if at least
        one judge survives, the query is graded on those votes. Only when *every*
        judge fails is the query a ``judge_error``.
        """
        if not item.system_response:
            return ItemResult(
                query_id=item.query_id,
                rubrics=item.rubrics,
                failed=True,
                failure_reason="no_response",
            )

        per_model: list[list[bool]] = []
        failed_checks_by_judge: dict[str, int] = {}
        for judge in self._judges:
            labels = self._judge_all_rubrics(judge, item, item.system_response)
            if labels is not None:
                per_model.append(labels)
            else:
                failed_checks_by_judge[judge.model] = failed_checks_by_judge.get(
                    judge.model, 0
                ) + len(item.rubrics)

        if not per_model:
            return ItemResult(
                query_id=item.query_id,
                rubrics=item.rubrics,
                failed=True,
                failure_reason="judge_error",
                failed_checks_by_judge=failed_checks_by_judge,
            )

        return ItemResult(
            query_id=item.query_id,
            rubrics=item.rubrics,
            labels=self.majority_vote(per_model),
            failed=False,
            failed_checks_by_judge=failed_checks_by_judge,
        )

    def _judge_all_rubrics(
        self, judge: Judge, item: EvalItem, response: str
    ) -> list[bool] | None:
        """Run one judge model over all of an item's rubrics, in batches.

        Returns labels aligned to ``item.rubrics``, or ``None`` if any batch
        errored or the model returned the wrong number of labels. A batch whose
        reply can't be parsed as JSON is dropped this way too (returns ``None``)
        rather than being silently scored as all-``False`` — otherwise garbage
        output would masquerade as a valid "nothing qualifies" vote.
        """
        labels: list[bool] = []
        rubrics = item.rubrics
        for start in range(0, len(rubrics), self._max_per_call):
            batch = rubrics[start : start + self._max_per_call]
            judgements = self._judge_batch(judge, item, response, batch)
            if judgements is None:
                return None
            for idx in range(len(batch)):
                entry = judgements.get(str(idx), {})
                label = entry.get("label", False) if isinstance(entry, dict) else False
                labels.append(bool(label))

        if len(labels) != len(rubrics):
            logger.error(
                "judge %s returned %d labels for %d rubrics on query %s",
                judge.model,
                len(labels),
                len(rubrics),
                item.query_id,
            )
            return None
        return labels

    def _judge_batch(
        self, judge: Judge, item: EvalItem, response: str, batch: list[Rubric]
    ) -> dict[str, dict] | None:
        """Run one judge call for a batch, re-prompting on unparseable JSON.

        Returns the parsed judgements (possibly empty), or ``None`` if the call
        errored or stayed unparseable after ``max_json_parse_retries`` retries.
        """
        user = RubricPrompt.build_user(item.query, item.query_date, response, batch)
        for attempt in range(self._max_json_parse_retries + 1):
            # Nudge the format on retries; the call is otherwise identical.
            user_prompt = (
                user
                if attempt == 0
                else user + RubricPrompt.JSON_FORMAT_RETRY_SUFFIX
            )
            try:
                text = judge.complete(RubricPrompt.SYSTEM, user_prompt)
            except Exception:
                logger.exception(
                    "judge %s failed on query %s", judge.model, item.query_id
                )
                return None
            judgements = RubricPrompt.try_parse(text)
            if judgements is not None:
                return judgements
            logger.warning(
                "judge %s returned unparseable JSON on query %s (attempt %d/%d)",
                judge.model,
                item.query_id,
                attempt + 1,
                self._max_json_parse_retries + 1,
            )
        return None

    @staticmethod
    def majority_vote(per_model_labels: list[list[bool]]) -> list[bool]:
        """Combine per-model rubric labels by majority vote.

        Each inner list is one model's labels, aligned across models. A rubric is
        qualified when a strict majority of models say so; an exact tie (only
        possible with an even number of votes) defers to the first model. Prefer
        an odd panel so ties never arise. Callers must drop failed (``None``)
        model results first.
        """
        if not per_model_labels:
            raise ValueError("majority_vote requires at least one model's labels")

        n = len(per_model_labels)
        width = len(per_model_labels[0])
        if any(len(labels) != width for labels in per_model_labels):
            raise ValueError("all models must return the same number of labels")

        result: list[bool] = []
        for i in range(width):
            votes = sum(labels[i] for labels in per_model_labels)
            if votes * 2 == n:  # tie -> first model decides
                result.append(per_model_labels[0][i])
            else:
                result.append(votes * 2 > n)
        return result
