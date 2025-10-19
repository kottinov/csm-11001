"""CLI for the reflection-based multi-agent coordination supervisor.

This uses LangGraph handoff tools for true reflection-based coordination as described
in the thesis, unlike the subprocess-based supervisor which runs agents in isolation.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from ..config import Settings, get_settings
from ..orchestrator.handoff_supervisor import default_query, run_supervisor_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RCA supervisor with reflection-based multi-agent coordination via LangGraph handoff tools."
    )
    parser.add_argument(
        "--question",
        help="The investigation question for the supervisor. If not provided, uses a default query based on configured datasets.",
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

    query = args.question or default_query(settings)

    diagnosis = run_supervisor_flow(query=query, settings=settings)

    print(diagnosis)


if __name__ == "__main__":
    main()


__all__ = ["main"]