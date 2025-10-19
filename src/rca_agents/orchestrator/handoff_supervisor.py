"""Reflection-based multi-agent coordination using LangGraph supervisor with handoff tools.

This implements the coordination protocol described in the thesis: agents use reflection
scores to assess their confidence and request collaboration from other specialized agents.
"""
from __future__ import annotations

from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..agents import log_agent, metrics_agent, report_agent
from ..config.settings import Settings, get_settings
from ..core.result import AgentResult


@tool
def investigate_logs(query: str) -> dict:
    """Investigate system logs using LATS-guided analysis.

    This agent examines log files to identify error patterns, timestamps,
    and causal relationships. Returns structured findings with confidence scores.
    """
    result: AgentResult = log_agent.run(query)
    return {
        "agent": "log_agent",
        "findings": result.summary,
        "confidence": result.confidence,
        "evidence": result.evidence,
        "reflections": result.reflections,
        "needs_metrics": result.needs_collaboration,
        "suggested_next": result.suggested_next_agent,
    }


@tool
def investigate_metrics(query: str, log_context: Optional[str] = None) -> dict:
    """Analyze metrics data, optionally with log findings context.

    This agent examines performance metrics (goroutines, memory, GC, load) to
    identify anomalies. If log_context is provided, it will assess correlation
    between metrics and log findings.
    """
    result: AgentResult = metrics_agent.run(query, log_context=log_context)
    return {
        "agent": "metrics_agent",
        "findings": result.summary,
        "confidence": result.confidence,
        "evidence": result.evidence,
        "reflections": result.reflections,
        "needs_collaboration": result.needs_collaboration,
    }


@tool
def assess_correlation(log_findings: str, metrics_findings: str, log_evidence: list[str], metrics_evidence: list[str]) -> str:
    """Use LLM to assess whether metrics findings correlate with log findings.

    This tool provides sophisticated correlation assessment based on the actual
    content of findings and evidence, rather than hardcoded heuristics.
    """
    settings = get_settings()
    llm = ChatAnthropic(model=settings.anthropic.supervisor_model)

    prompt = f"""Assess the correlation between log analysis and metrics analysis findings.

    Log Findings:
    {log_findings}

    Log Evidence:
    {chr(10).join(f"- {e}" for e in log_evidence)}

    Metrics Findings:
    {metrics_findings}

    Metrics Evidence:
    {chr(10).join(f"- {e}" for e in metrics_evidence)}

    Determine if the metrics findings support, contradict, or are unrelated to the log findings.
    Respond with one of:
    - "strong_correlation": Metrics clearly confirm or explain the log findings
    - "weak_correlation": Some connection exists but not definitive
    - "no_correlation": Metrics appear normal despite log errors
    - "contradictory": Metrics contradict what logs suggest

    Provide your assessment and brief reasoning."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def generate_report(log_findings_dict: dict, metrics_findings_dict: dict, output_path: str = "reports/incident_report.tex") -> str:
    """Generate a professional LaTeX incident report from investigation findings.

    This agent uses ReAct pattern with file editing tools to create, review, edit,
    and compile a publication-ready PDF report.

    Args:
        log_findings_dict: Result dictionary from investigate_logs tool
        metrics_findings_dict: Result dictionary from investigate_metrics tool
        output_path: Where to write the .tex file (default: reports/incident_report.tex)

    Returns:
        Status message indicating PDF location or compilation errors
    """
    log_result = AgentResult(
        agent_name=log_findings_dict["agent"],
        summary=log_findings_dict["findings"],
        confidence=log_findings_dict["confidence"],
        evidence=log_findings_dict["evidence"],
        reflections=log_findings_dict["reflections"],
        needs_collaboration=log_findings_dict["needs_metrics"],
        suggested_next_agent=log_findings_dict.get("suggested_next"),
    )

    metrics_result = AgentResult(
        agent_name=metrics_findings_dict["agent"],
        summary=metrics_findings_dict["findings"],
        confidence=metrics_findings_dict["confidence"],
        evidence=metrics_findings_dict["evidence"],
        reflections=metrics_findings_dict["reflections"],
        needs_collaboration=metrics_findings_dict["needs_collaboration"],
    )

    return report_agent.run(log_result, metrics_result, output_path)


def build_supervisor(settings: Optional[Settings] = None) -> object:
    """Create the LangGraph supervisor agent with handoff tools.

    The supervisor coordinates specialized agents using their reflection scores
    to make collaboration decisions.
    """
    settings = settings or get_settings()
    llm = ChatAnthropic(model=settings.anthropic.supervisor_model)

    system_prompt = """You are an RCA (Root Cause Analysis) coordinator managing specialized agents.

    Your workflow:
    1. Call investigate_logs to analyze system logs
    2. If log confidence < 0.7 OR user requests comprehensive analysis:
       Call investigate_metrics for performance data
    3. If you have both log and metrics findings:
       Call assess_correlation to check if findings align
    4. MANDATORY: Call generate_report with log_findings_dict and metrics_findings_dict
       to create the final LaTeX/PDF incident report
    5. After report is generated, provide the PDF path to the user

    Guidelines:
    - Always call generate_report when you have investigation findings
    - The report tool will create a publication-ready PDF document
    - Do NOT synthesize findings yourself - let generate_report handle that
    - Your final response should include the PDF path from generate_report

    Available tools:
    - investigate_logs: Analyze system logs for errors and patterns
    - investigate_metrics: Examine performance metrics for anomalies
    - assess_correlation: Determine if metrics support log findings
    - generate_report: MUST CALL THIS to create the final LaTeX/PDF report"""

    return create_react_agent(
        llm,
        tools=[investigate_logs, investigate_metrics, assess_correlation, generate_report],
        prompt=system_prompt,
    )


def run_supervisor_flow(query: str, settings: Optional[Settings] = None) -> str:
    """Execute the reflection-based multi-agent coordination flow.

    Args:
        query: The RCA investigation question
        settings: Optional settings override

    Returns:
        Final synthesized diagnosis from the supervisor
    """
    settings = settings or get_settings()
    supervisor = build_supervisor(settings)

    result = supervisor.invoke({"messages": [HumanMessage(content=query)]})

    final_message = result["messages"][-1]
    return final_message.content


def default_query(settings: Optional[Settings] = None) -> str:
    """Generate default investigation query based on configured datasets."""
    settings = settings or get_settings()
    return (
        f"Investigate the root cause of 401 authentication errors. "
        f"Analyze logs in '{settings.dataset.logs_root}' and "
        f"metrics in '{settings.dataset.metrics_csv}'. "
        f"Compare the error scenario 'access_token_auth_header_error_401' with baseline 'correct'."
    )


__all__ = ["build_supervisor", "run_supervisor_flow", "default_query"]
