from __future__ import annotations

import subprocess
import sys
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from ..agents.log_agent import default_question as default_log_question
from ..agents.metrics_agent import default_question as default_metrics_question
from ..agents import report_agent
from ..core.result import AgentResult
from ..config.settings import Settings, get_settings
import json

def _run_agent_module(module: str, cli_args: list[str], timeout: int = 600) -> str:
    result = subprocess.run(
        [sys.executable, "-m", module, *cli_args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"{module} failed (code {result.returncode}): {stderr}"
        )

    return result.stdout.strip()


def run_log_agent_isolated(
    query: str,
    timeout: int = 1800,
    settings: Optional[Settings] = None,
) -> str:
    """Execute the log agent in a subprocess and return its summary."""
    cli_args = ["--no-stream", "--question", query]
    if settings:
        cli_args.extend(["--logs-root", str(settings.dataset.logs_root)])
    return _run_agent_module("rca_agents.cli.log_agent", cli_args, timeout)


def run_metrics_agent_isolated(
    query: str,
    timeout: int = 1800,
    settings: Optional[Settings] = None,
) -> str:
    """Execute the metrics agent in a subprocess and return its summary."""
    cli_args = ["--no-stream", "--question", query]
    if settings:
        cli_args.extend(
            [
                "--metrics-csv",
                str(settings.dataset.metrics_csv),
                "--charts-dir",
                str(settings.dataset.metrics_chart_dir),
            ]
        )
    return _run_agent_module("rca_agents.cli.metrics_agent", cli_args, timeout)


def synthesize_findings(
    log_summary: str,
    metrics_summary: str,
    original_query: str,
    settings: Optional[Settings] = None,
) -> str:
    """Use a lightweight Claude call to synthesize agent summaries."""
    settings = settings or get_settings()
    llm = ChatAnthropic(model=settings.anthropic.supervisor_model)

    synthesis_prompt = f"""You are a Root Cause Analysis (RCA) supervisor. Two specialized agents have completed their work.

    Original Question:
    {original_query}

    LOG ANALYSIS SUMMARY:
    {log_summary}

    METRICS ANALYSIS SUMMARY:
    {metrics_summary}

    Synthesize these findings into a concise RCA report that:
    1. Identifies the primary root cause based on evidence
    2. Explains whether the log errors correlate with metrics anomalies
    3. Distinguishes application errors from infrastructure issues
    4. Notes when metrics remain stable despite log errors
    5. Provides actionable recommendations rooted in the evidence
    6. Highlights any gaps or conflicting observations.

    Do not invent correlations; report honestly based on the provided summaries."""

    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    return getattr(response, "content", str(response))


def run_supervisor_flow(
    log_query: Optional[str] = None,
    metrics_query: Optional[str] = None,
    original_query: Optional[str] = None,
    settings: Optional[Settings] = None,
    generate_report: bool = True,
) -> str:
    """Run both agents in isolation, synthesize their findings, and optionally generate a PDF report."""
    settings = settings or get_settings()

    resolved_log_query = log_query or default_log_question(settings)
    resolved_metrics_query = metrics_query or default_metrics_question(settings)
    resolved_original_query = (
        original_query
        or "Perform a comprehensive Root Cause Analysis using both log and metrics perspectives."
    )

    # Run log and metrics agents
    log_output = run_log_agent_isolated(resolved_log_query, settings=settings)
    metrics_output = run_metrics_agent_isolated(resolved_metrics_query, settings=settings)

    # Parse agent results
    try:
        log_result = AgentResult(**eval(log_output.replace("AgentResult(", "dict(").replace(")", ")")))
    except:
        # Fallback if parsing fails
        log_result = AgentResult(
            agent_name="log_agent",
            summary=log_output,
            confidence=0.7,
            evidence=[],
            reflections="",
            needs_collaboration=False
        )

    try:
        metrics_result = AgentResult(**eval(metrics_output.replace("AgentResult(", "dict(").replace(")", ")")))
    except:
        metrics_result = AgentResult(
            agent_name="metrics_agent",
            summary=metrics_output,
            confidence=0.7,
            evidence=[],
            reflections="",
            needs_collaboration=False
        )

    # Generate synthesis
    synthesis = synthesize_findings(
        log_result.summary,
        metrics_result.summary,
        resolved_original_query,
        settings=settings,
    )

    # Generate PDF report if requested
    if generate_report:
        try:
            report_path = report_agent.run(
                log_result=log_result,
                metrics_result=metrics_result,
                output_path="reports/incident_report.tex",
                settings=settings,
            )
            synthesis += f"\n\n---\nPDF Report generated: {report_path}"
        except Exception as e:
            synthesis += f"\n\n---\nReport generation failed: {str(e)}"

    return synthesis


__all__ = [
    "run_log_agent_isolated",
    "run_metrics_agent_isolated",
    "synthesize_findings",
    "run_supervisor_flow",
]
