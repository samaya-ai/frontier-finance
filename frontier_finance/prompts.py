"""Judge prompt construction and response parsing.

The judge is shown the query, its date, the response under evaluation, and a
0-indexed list of rubrics, and must return a JSON object keyed by rubric index
with a boolean ``label`` per rubric.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar

from frontier_finance.models import Rubric

logger = logging.getLogger(__name__)


class RubricPrompt:
    """Builds the judge prompt for a batch of rubrics and parses the reply."""

    # Appended to the user prompt when re-prompting a judge after an unparseable
    # reply. Changing the prompt matters even at temperature 0, where an
    # unchanged prompt would deterministically reproduce the same bad output.
    JSON_FORMAT_RETRY_SUFFIX: ClassVar[str] = (
        "\n\nIMPORTANT: Make sure you strictly follow the output format and output a valid "
        "JSON object that can be parsed successfully, and nothing else."
    )

    SYSTEM: ClassVar[str] = """\
You are a senior financial analyst. Your task is to evaluate a financial report against a list of pre-defined rubrics. The report presented to you is generated to answer a specific financial query. For each given rubric, you are expected to produce a binary judgement on whether the rubric is satisfied or not by the financial report.

For each task, you will be given the following:
1. A financial query, which specifies the information the user is seeking.
2. The date the query was made. This is important for assessing the time understanding of the system. Whenever necessary, you should use this date as the temporal anchor for interpreting relative date terms in both the query and the rubrics.
3. A financial report which aims to answer that query.
4. One or more natural language rubrics, each checking a specific aspect of the report.

All of the input will be clearly marked in XML tags. Your task is to judge whether the report adequately satisfies each of the given rubrics. You must evaluate the report objectively and thoroughly.

Pay special attention to the following aspects when making your judgement:
1. **Each rubric should be judged independently**. Even in the case that one rubric seems related to another, you need to give your judgement of whether each rubric is satisfied independently.
2. **Pay attention to numerical units**. The report and the rubric might use different units to represent the same number. Take this into account when making your judgement. For example, "USD 2.1 billion" is equivalent to "USD 2,100 million".
3. **Accept reasonable numerical approximation**. A figure in the report is acceptable if it equals the rubric's figure after rounding the rubric's figure to the (coarser) precision the report uses. A figure stated at the same or finer precision than the rubric's, but with a different value, is NOT acceptable — even if numerically close. For example, against a rubric value of "3,098 million": "3.1 billion" is acceptable (a correct rounding to two significant figures), but "3,105 million" is not (it asserts a precise, different value). Likewise against "7.14%": "7.1%" is acceptable, but "7.25%" is not.\
"""

    _TEMPLATE: ClassVar[str] = """\
You will evaluate the report below against the given set of rubrics. The report has been written to answer a specific query.

The query is provided below within the <query> tags.
<query>
{query}
</query>

The date the query was submitted is provided below within the <date> tags. This is important for assessing whether the report correctly understands the time aspect of the query.
<date>
{query_date}
</date>

The financial report is provided below within the <report> tags.
<report>
{generated_summary}
</report>

Now that you have read the query and the report, please evaluate whether the report satisfies each of the following rubrics. The list of rubrics is provided below within the <rubrics> tags. Each rubric is annotated with a unique ID, which you should use in your output to refer to that rubric.
<rubrics>
{criteria}
</rubrics>

For each rubric, determine if the report adequately satisfies it. As a reminder, pay attention to the following aspects mentioned before:
- Each rubric should be judged independently.
- Pay attention to numerical units.
- Accept reasonable numerical approximation.

Your output must be ONLY a valid JSON object with the following structure:
```json
{{
  "0": {{
    "reason": "concise 1-sentence reason for your judgement on rubric 0",
    "label": true/false
  }},
  "1": {{
    "reason": "concise 1-sentence reason for your judgement on rubric 1",
    "label": true/false
  }},
  ...
}}
```

The keys must be string representations of the given rubric ID (starting from 0).
The "reason" field should contain your 1-sentence concise reasoning about whether the rubric is satisfied.
The "label" field must be a boolean value (true if the rubric is satisfied, false otherwise).

Now provide your judgements. Recall that the user query is: <query> {query} </query> and the date the query was made is: <date> {query_date} </date>.

Output your evaluation as the JSON object specified above and nothing else.\
"""

    _JSON_BLOCK_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL
    )

    @classmethod
    def build_user(
        cls,
        query: str,
        query_date: str,
        system_response: str,
        batch_rubrics: list[Rubric],
    ) -> str:
        """Render the full user prompt for one batch of rubrics (0-indexed within the batch)."""
        criteria = "\n\n".join(
            f"{i}. {r.rubric_text}".strip() for i, r in enumerate(batch_rubrics)
        )
        return cls._TEMPLATE.format(
            query=query,
            query_date=query_date,
            generated_summary=system_response,
            criteria=criteria,
        )

    @classmethod
    def try_parse(cls, response_str: str) -> dict[str, dict[str, Any]] | None:
        """Parse the judge's JSON response into ``{index: {"reason", "label"}}``.

        Tolerates ```json fenced blocks and prose around the JSON. Returns the
        parsed object (possibly empty) on success, or ``None`` if no JSON object
        can be recovered — so callers can tell a genuine parse failure apart from
        a model that legitimately judged nothing as qualifying.
        """
        if not response_str:
            logger.warning("empty judge response")
            return None

        for match in cls._JSON_BLOCK_RE.findall(response_str):
            try:
                parsed = json.loads(match.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        start, end = response_str.find("{"), response_str.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(response_str[start : end + 1])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed

        logger.error("could not extract JSON from judge response: %.200s", response_str)
        return None
