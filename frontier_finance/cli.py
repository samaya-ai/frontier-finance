from __future__ import annotations

import argparse
import dataclasses
import logging

from frontier_finance.config import EvalConfig
from frontier_finance.runner import EvalRunner


class CLI:
    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="frontier-finance-grader",
            description="Grade system responses against per-query rubrics with an LLM judge panel.",
        )
        parser.add_argument(
            "--config", required=True, help="Path to the YAML run config."
        )
        parser.add_argument("--rubrics", help="Override rubrics_path from the config.")
        parser.add_argument(
            "--responses", help="Override responses_path from the config."
        )
        parser.add_argument("--output-dir", help="Override output_dir from the config.")
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Enable debug logging."
        )
        return parser

    @classmethod
    def main(cls, argv: list[str] | None = None) -> int:
        args = cls._build_parser().parse_args(argv)

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        cfg = EvalConfig.load(args.config)
        overrides = {
            k: v
            for k, v in (
                ("rubrics_path", args.rubrics),
                ("responses_path", args.responses),
                ("output_dir", args.output_dir),
            )
            if v is not None
        }
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)

        EvalRunner(cfg).run()
        return 0


if __name__ == "__main__":
    raise SystemExit(CLI.main())
