from __future__ import annotations

import argparse
import os
from dataclasses import replace
import json
from pathlib import Path
from typing import Iterable, Optional

from ..agents.log_agent import (
    DEFAULT_LOG_QUESTION,
    build_graph,
    default_question,
    run as run_log_agent,
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
    parser = argparse.ArgumentParser(description="Run the log RCA agent.")
    parser.add_argument("--question", help="RCA question for the log agent.")
    parser.add_argument(
        "--logs-root",
        help="Override the root directory containing log data for this investigation.",
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

    if args.logs_root:
        logs_root = Path(args.logs_root).expanduser().resolve()
        settings = replace(
            settings,
            dataset=replace(settings.dataset, logs_root=logs_root),
        )
        os.environ["RCA_LOGS_ROOT"] = str(logs_root)

    question = args.question or default_question(settings)
    graph = build_graph(settings)

    if args.no_stream:
        result = run_log_agent(question, graph)
        print(json.dumps(result.to_dict(), indent=2))
        return

    _stream_steps(graph, question)


__all__ = ["main"]


if __name__ == "__main__":
    main()
