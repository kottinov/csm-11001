from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from ..config import Settings, get_settings
from ..orchestrator import run_supervisor_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run log and metrics RCA agents in isolation and synthesize their findings."
    )
    parser.add_argument("--log-question", help="Question provided to the log agent.")
    parser.add_argument(
        "--metrics-question", help="Question provided to the metrics agent."
    )
    parser.add_argument(
        "--summary-question",
        help="Prompt passed to the supervisor synthesizer.",
    )
    parser.add_argument(
        "--logs-root",
        help="Override the root directory containing log data for this investigation.",
    )
    parser.add_argument(
        "--metrics-csv",
        help="Override the metrics CSV file path used during analysis.",
    )
    parser.add_argument(
        "--charts-dir",
        help="Directory to write generated comparison charts into.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings: Settings = get_settings()

    if args.logs_root:
        logs_root = Path(args.logs_root).expanduser().resolve()
        settings = replace(
            settings,
            dataset=replace(settings.dataset, logs_root=logs_root),
        )
        os.environ["RCA_LOGS_ROOT"] = str(logs_root)

    if args.metrics_csv:
        metrics_csv = Path(args.metrics_csv).expanduser().resolve()
        settings = replace(
            settings,
            dataset=replace(settings.dataset, metrics_csv=metrics_csv),
        )
        os.environ["RCA_METRICS_CSV"] = str(metrics_csv)

    if args.charts_dir:
        charts_dir = Path(args.charts_dir).expanduser().resolve()
        settings = replace(
            settings,
            dataset=replace(settings.dataset, metrics_chart_dir=charts_dir),
        )
        os.environ["RCA_METRICS_CHART_DIR"] = str(charts_dir)

    log_query = args.log_question
    metrics_query = args.metrics_question
    summary_question = args.summary_question

    synthesis = run_supervisor_flow(
        log_query=log_query,
        metrics_query=metrics_query,
        original_query=summary_question,
        settings=settings,
    )

    print(synthesis)


__all__ = ["main"]


if __name__ == "__main__":
    main()
