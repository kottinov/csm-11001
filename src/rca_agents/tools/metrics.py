from __future__ import annotations

import os
import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from typing import Dict, List

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from langchain_core.tools import tool

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

DEFAULT_OUTPUT_DIR = Path(os.getenv("RCA_METRICS_CHART_DIR", "charts"))


def _load_dataframe(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load metrics CSV '{csv_path}': {exc}") from exc


@tool
def load_metrics_csv(csv_path: str) -> str:
    """Return available scenario counts for a metrics CSV file."""
    try:
        df = _load_dataframe(csv_path)
    except RuntimeError as exc:
        return f"Error loading CSV: {exc}"

    scenarios = df["test_name"].value_counts()
    lines = [f"Loaded metrics with {len(df)} total rows", "", "Available scenarios:"]
    for scenario, count in scenarios.items():
        lines.append(f"  - {scenario}: {count} data points")
    return "\n".join(lines)


def _compare_scenarios(df: pd.DataFrame, error_scenario: str, baseline: str) -> Dict[str, Dict[str, float]]:
    baseline_data = df[df["test_name"] == baseline]
    error_data = df[df["test_name"] == error_scenario]

    if baseline_data.empty:
        return {"error": f"Baseline scenario '{baseline}' not found"}
    if error_data.empty:
        return {"error": f"Error scenario '{error_scenario}' not found"}

    return {
        "scenario_info": {
            "error_scenario": error_scenario,
            "baseline": baseline,
            "error_data_points": int(len(error_data)),
            "baseline_data_points": int(len(baseline_data)),
        },
        "goroutines": {
            "baseline_avg": float(baseline_data["go_goroutines"].mean()),
            "error_avg": float(error_data["go_goroutines"].mean()),
            "baseline_max": float(baseline_data["go_goroutines"].max()),
            "error_max": float(error_data["go_goroutines"].max()),
        },
        "memory": {
            "baseline_avg_available": float(baseline_data["node_memory_MemAvailable_bytes"].mean()),
            "error_avg_available": float(error_data["node_memory_MemAvailable_bytes"].mean()),
            "baseline_avg_heap": float(baseline_data["go_memstats_heap_alloc_bytes"].mean()),
            "error_avg_heap": float(error_data["go_memstats_heap_alloc_bytes"].mean()),
        },
        "gc_duration": {
            "baseline_p50": float(baseline_data["go_gc_duration_seconds&quantile=0.5"].mean()),
            "error_p50": float(error_data["go_gc_duration_seconds&quantile=0.5"].mean()),
            "baseline_p99": float(baseline_data["go_gc_duration_seconds&quantile=1"].mean()),
            "error_p99": float(error_data["go_gc_duration_seconds&quantile=1"].mean()),
        },
        "system_load": {
            "baseline_load1": float(baseline_data["node_load1"].mean()),
            "error_load1": float(error_data["node_load1"].mean()),
            "baseline_load5": float(baseline_data["node_load5"].mean()),
        },
    }


@tool
def compare_scenarios_metrics(csv_path: str, error_scenario: str, baseline: str = "correct") -> Dict[str, Dict[str, float]]:
    """Compare key metrics between an error scenario and a baseline scenario."""
    try:
        df = _load_dataframe(csv_path)
    except RuntimeError as exc:
        return {"error": str(exc)}

    return _compare_scenarios(df, error_scenario, baseline)


def _ensure_output_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create output directory '{path}': {exc}") from exc


def _plot_bar(ax, labels: List[str], values: List[float], title: str, ylabel: str, colors: List[str], annotations: List[str]) -> None:
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    for bar, annotation in zip(bars, annotations):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            annotation,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    ax.grid(True, alpha=0.3, axis="y")


@tool
def generate_comparison_charts(
    csv_path: str,
    error_scenario: str,
    baseline: str = "correct",
) -> str:
    """Generate comparison charts between baseline and error scenarios.

    Charts are always generated in the configured charts directory.
    """
    directory = DEFAULT_OUTPUT_DIR
    try:
        _ensure_output_dir(directory)
    except RuntimeError as exc:
        return f"Error generating charts: {exc}"

    try:
        df = _load_dataframe(csv_path)
    except RuntimeError as exc:
        return f"Error generating charts: {exc}"

    baseline_data = df[df["test_name"] == baseline].copy()
    error_data = df[df["test_name"] == error_scenario].copy()
    if baseline_data.empty or error_data.empty:
        return "Error: One or both scenarios not found in data"

    charts_created: List[Path] = []

    baseline_data["index"] = range(len(baseline_data))
    error_data["index"] = range(len(error_data))

    fig, ax = plt.subplots(figsize=(10, 7))
    data_to_plot = [baseline_data["go_goroutines"], error_data["go_goroutines"]]
    parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2196F3")
        pc.set_alpha(0.6)
    ax.boxplot(
        data_to_plot,
        positions=[1, 2],
        widths=0.3,
        boxprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=2),
        capprops=dict(color="black", linewidth=2),
        medianprops=dict(color="red", linewidth=2),
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "Error"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Goroutines", fontsize=13, fontweight="bold")
    ax.set_title(f"Goroutines Distribution\n{error_scenario}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    chart_path = directory / "1_goroutines_distribution.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    charts_created.append(chart_path)

    fig, ax = plt.subplots(figsize=(10, 7))
    mem_baseline = baseline_data["node_memory_MemAvailable_bytes"].mean() / 1e9
    mem_error = error_data["node_memory_MemAvailable_bytes"].mean() / 1e9
    mem_change = ((mem_error - mem_baseline) / mem_baseline * 100) if mem_baseline else 0
    _plot_bar(
        ax,
        ["Baseline", "Error"],
        [mem_baseline, mem_error],
        f"Memory Available Comparison\n{mem_change:+.1f}% change",
        "Available Memory (GB)",
        ["#4CAF50", "#F44336"],
        [f"{mem_baseline:.2f} GB", f"{mem_error:.2f} GB"],
    )
    chart_path = directory / "2_memory_available.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    charts_created.append(chart_path)

    fig, ax = plt.subplots(figsize=(10, 7))
    heap_baseline = baseline_data["go_memstats_heap_alloc_bytes"].mean() / 1e6
    heap_error = error_data["go_memstats_heap_alloc_bytes"].mean() / 1e6
    heap_change = ((heap_error - heap_baseline) / heap_baseline * 100) if heap_baseline else 0
    _plot_bar(
        ax,
        ["Baseline", "Error"],
        [heap_baseline, heap_error],
        f"Heap Allocation Comparison\n{heap_change:+.1f}% change",
        "Heap Allocated (MB)",
        ["#4CAF50", "#F44336"],
        [f"{heap_baseline:.2f} MB", f"{heap_error:.2f} MB"],
    )
    chart_path = directory / "3_heap_allocation.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    charts_created.append(chart_path)

    fig, ax = plt.subplots(figsize=(12, 7))
    x_labels = ["1min", "5min", "15min"]
    baseline_loads = [
        baseline_data["node_load1"].mean(),
        baseline_data["node_load5"].mean(),
        baseline_data["node_load15"].mean(),
    ]
    error_loads = [
        error_data["node_load1"].mean(),
        error_data["node_load5"].mean(),
        error_data["node_load15"].mean(),
    ]
    x_pos = np.arange(len(x_labels))
    width = 0.35
    ax.bar(x_pos - width / 2, baseline_loads, width, label="Baseline", color="#4CAF50", alpha=0.8, edgecolor="black", linewidth=2)
    ax.bar(x_pos + width / 2, error_loads, width, label="Error", color="#F44336", alpha=0.8, edgecolor="black", linewidth=2)
    for index, (base, err) in enumerate(zip(baseline_loads, error_loads)):
        if base:
            change = (err - base) / base * 100
            ax.text(
                x_pos[index],
                max(base, err) + 0.5,
                f"{change:+.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )
    ax.set_ylabel("System Load", fontsize=13, fontweight="bold")
    ax.set_title("System Load Comparison (1min, 5min, 15min averages)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    chart_path = directory / "4_system_load.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    charts_created.append(chart_path)

    result_lines = [f"Successfully generated {len(charts_created)} focused charts:"]
    result_lines.extend(f"  {index}. {path}" for index, path in enumerate(charts_created, start=1))
    return "\n".join(result_lines)


@tool
def get_metrics_summary(csv_path: str, scenario: str) -> Dict[str, Dict[str, float]]:
    """Return summary statistics for a specific test scenario."""
    try:
        df = _load_dataframe(csv_path)
    except RuntimeError as exc:
        return {"error": str(exc)}

    data = df[df["test_name"] == scenario]
    if data.empty:
        return {"error": f"Scenario '{scenario}' not found"}

    return {
        "scenario": scenario,
        "data_points": int(len(data)),
        "goroutines": {
            "mean": float(data["go_goroutines"].mean()),
            "min": float(data["go_goroutines"].min()),
            "max": float(data["go_goroutines"].max()),
            "std": float(data["go_goroutines"].std()),
        },
        "heap_allocated_mb": {
            "mean": float(data["go_memstats_heap_alloc_bytes"].mean() / 1e6),
            "min": float(data["go_memstats_heap_alloc_bytes"].min() / 1e6),
            "max": float(data["go_memstats_heap_alloc_bytes"].max() / 1e6),
        },
        "system_load": {
            "load1": float(data["node_load1"].mean()),
            "load5": float(data["node_load5"].mean()),
            "load15": float(data["node_load15"].mean()),
        },
    }


METRICS_TOOLS = [
    load_metrics_csv,
    compare_scenarios_metrics,
    generate_comparison_charts,
    get_metrics_summary,
]

__all__ = [
    "METRICS_TOOLS",
    "load_metrics_csv",
    "compare_scenarios_metrics",
    "generate_comparison_charts",
    "get_metrics_summary",
]
