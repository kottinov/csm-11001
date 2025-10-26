"""CLI for React log analysis agent (baseline)."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from ..react_agents.react_log_agent import (
    DEFAULT_LOG_QUESTION,
    build_graph,
    default_question,
    run as run_react_log_agent,
)
from ..config import Settings, get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the React log RCA agent (baseline)."
    )
    parser.add_argument("--question", help="RCA question for the log agent.")
    parser.add_argument(
        "--logs-root",
        help="Override the root directory containing log data for this investigation.",
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

    result = run_react_log_agent(question, graph)
    print(result)


__all__ = ["main"]


if __name__ == "__main__":
    main()