from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from ..agents.metrics_agent import (
    DEFAULT_METRICS_QUESTION,
    build_graph,
    default_question,
    run as run_metrics_agent,
)
from ..config import Settings, get_settings


def _stream_steps(graph, question: str) -> None:
    last_step = None
    for step in graph.stream({"input": question}):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        root = step_state["root"]
        print(f"Step: {step_name}")
        print(f"Tree height: {root.height}")
        print("---")

    if last_step:
        solution_node = last_step[next(iter(last_step))]["root"].best_solution()
        trajectory = solution_node.get_trajectory(include_reflections=False)
        final_message = trajectory[-1]
      
        print(getattr(final_message, "content", str(final_message)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the metrics RCA agent.")
    parser.add_argument("--question", help="RCA question for the metrics agent.")
    parser.add_argument(
        "--metrics-csv",
        help="Override the metrics CSV file path used during analysis.",
    )
    parser.add_argument(
        "--charts-dir",
        help="Directory to write generated comparison charts into.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable step-by-step streaming and only print the final answer.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings: Settings = get_settings()

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

    question = args.question or default_question(settings)
    graph = build_graph(settings)

    if args.no_stream:
        print(run_metrics_agent(question, graph))
    else:
        _stream_steps(graph, question)


__all__ = ["main"]


if __name__ == "__main__":
    main()
