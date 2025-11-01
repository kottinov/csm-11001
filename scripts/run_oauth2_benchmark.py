#!/usr/bin/env python3
"""Run the OAuth2 benchmark scenarios described in the thesis.

This utility executes the log and metrics agents across the normalized
Light OAuth2 dataset, capturing structured outputs that can be used to
reproduce the evaluation tables in Chapter 8.

Example:
    python scripts/run_oauth2_benchmark.py \
        --dataset-root lo2-sample \
        --timestamp 1719592986 \
        --output outputs/oauth2_benchmark_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import time
from typing import Iterable

from rca_agents.agents import log_agent, metrics_agent
from rca_agents.config import Settings, get_settings
from rca_agents.core.result import AgentResult


def _load_ground_truth(dataset_root: Path) -> dict:
    ground_truth_path = dataset_root / "ground_truth_verified.json"
    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Expected ground truth file at {ground_truth_path}. "
            "Ensure the OAuth2 dataset has been normalized as described in the thesis."
        )
    return json.loads(ground_truth_path.read_text(encoding="utf-8"))


def _scenario_dirs(dataset_root: Path, timestamp: str) -> Iterable[Path]:
    log_root = dataset_root / "logs" / f"light-oauth2-data-{timestamp}"
    if not log_root.exists():
        raise FileNotFoundError(
            f"Log directory {log_root} not found. "
            "Use --timestamp to point to an available capture."
        )
    return sorted(path for path in log_root.iterdir() if path.is_dir())


def _configure_settings(base: Settings, logs_root: Path, metrics_csv: Path, charts_dir: Path) -> Settings:
    """Create a Settings instance with scenario-specific dataset paths."""
    dataset = replace(
        base.dataset,
        logs_root=logs_root,
        metrics_csv=metrics_csv,
        metrics_chart_dir=charts_dir,
    )
    return replace(base, dataset=dataset)


def _run_agents_for_scenario(scenario: Path, settings: Settings) -> tuple[AgentResult, AgentResult, float]:
    """Execute both agents for a given scenario and return their results plus runtime."""
    log_graph = log_agent.build_graph(settings)
    metrics_graph = metrics_agent.build_graph(settings)

    question_log = log_agent.default_question(settings)
    question_metrics = metrics_agent.default_question(settings)

    started = time.perf_counter()
    log_result = log_agent.run(question_log, log_graph)
    metrics_result = metrics_agent.run(question_metrics, metrics_graph)
    elapsed = time.perf_counter() - started

    return log_result, metrics_result, elapsed


def _write_result(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(payload))
        sink.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OAuth2 RCA benchmark scenarios.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("lo2-sample"),
        help="Root directory containing the normalized OAuth2 dataset (default: lo2-sample).",
    )
    parser.add_argument(
        "--timestamp",
        default="1719592986",
        help="Timestamp suffix identifying the capture to use under logs/ and metrics/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/oauth2_benchmark_results.jsonl"),
        help="Where to append JSONL results (default: outputs/oauth2_benchmark_results.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of scenarios to run (0 = all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which scenarios would run without invoking the agents.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    ground_truth = _load_ground_truth(dataset_root)

    scenario_dirs = _scenario_dirs(dataset_root, args.timestamp)
    metrics_csv = dataset_root / "metrics" / f"light-oauth2-data-{args.timestamp}.csv"
    charts_dir = dataset_root / "charts"

    base_settings = get_settings()

    executed = 0
    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        if args.limit and executed >= args.limit:
            break

        scenario_settings = _configure_settings(base_settings, scenario_dir, metrics_csv, charts_dir)

        if args.dry_run:
            print(f"[DRY RUN] Would execute scenario '{scenario_name}' using logs at {scenario_dir}")
            executed += 1
            continue

        log_result, metrics_result, elapsed = _run_agents_for_scenario(scenario_dir, scenario_settings)

        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "scenario": scenario_name,
            "ground_truth": ground_truth.get(scenario_name, {}),
            "log_agent": log_result.to_dict(),
            "metrics_agent": metrics_result.to_dict(),
            "runtime_seconds": elapsed,
        }
        _write_result(args.output, payload)

        executed += 1
        print(f"Completed {scenario_name} in {elapsed:.2f}s – results appended to {args.output}")

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Finished {executed} scenarios. Results available at {args.output}")


if __name__ == "__main__":
    main()
