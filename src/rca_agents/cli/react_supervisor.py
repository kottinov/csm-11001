"""CLI for React multi-agent supervisor (baseline)."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from ..react_agents.react_supervisor import (
    build_supervisor_graph,
    default_supervisor_question,
    run_supervisor,
)
from ..config import Settings, get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the React multi-agent supervisor (baseline)."
    )
    parser.add_argument(
        "--question", help="RCA question for the multi-agent supervisor."
    )
    parser.add_argument(
        "--logs-root",
        help="Override the root directory containing log data for this investigation.",
    )
    parser.add_argument(
        "--metrics-csv",
        help="Override the path to the metrics CSV file.",
    )
    parser.add_argument(
        "--charts-dir",
        help="Override the output directory for comparison charts.",
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

    question = args.question or default_supervisor_question(settings)
    graph = build_supervisor_graph(settings)

    result = run_supervisor(question, graph, settings)
    print(result)


__all__ = ["main"]


if __name__ == "__main__":
    main()