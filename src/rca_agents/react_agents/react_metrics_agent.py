"""React agent for metrics analysis - baseline implementation."""
from __future__ import annotations

from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from ..config.settings import Settings, get_settings
from ..core.result import AgentResult
from ..tools.metrics import METRICS_TOOLS

DEFAULT_METRICS_QUESTION = (
    "Analyze the metrics captured in the configured dataset, compare the error scenario with the baseline, "
    "and identify anomalies that could explain the authentication errors. Generate comparison charts."
)


def _build_system_prompt(settings: Settings) -> str:
    """Build the system prompt for the React metrics agent."""
    return (
        "You are a metrics analysis expert supporting Root Cause Analysis (RCA). "
        "Investigate performance metrics data to understand incidents.\n\n"
        "Available tools:\n"
        "- load_metrics_csv: Inspect scenarios and counts\n"
        "- get_metrics_summary: Summarize metrics for a scenario\n"
        "- compare_scenarios_metrics: Contrast error vs baseline metrics\n"
        "- generate_comparison_charts: Visualize differences\n\n"
        f"Dataset location:\n"
        f"- The metrics CSV for this investigation is located at: {settings.dataset.metrics_csv}\n\n"
        "Guidelines:\n"
        "1. Load the CSV to understand scenarios and sample sizes\n"
        "2. Compare the error scenario with the baseline 'correct'\n"
        "3. Assess goroutines, memory, GC, and system load metrics\n"
        "4. Highlight statistically meaningful differences; acknowledge normal findings\n"
        "5. Generate charts when requested, noting output locations\n"
        "6. Report findings honestly, especially when metrics look nominal."
    )


def build_graph(settings: Optional[Settings] = None):
    """Create the React agent graph for metrics analysis.

    This is a baseline implementation using LangGraph's create_react_agent
    for comparison with the LATS-based approach.
    """
    settings = settings or get_settings()
    llm = ChatAnthropic(model=settings.anthropic.metrics_model)

    system_prompt = _build_system_prompt(settings)

    graph = create_react_agent(
        llm,
        tools=METRICS_TOOLS,
        prompt=system_prompt,
    )

    return graph


def run(
    question: str = DEFAULT_METRICS_QUESTION,
    graph=None,
    log_context: Optional[str] = None,
) -> AgentResult:
    """Execute the React metrics analysis agent and return structured result.

    Args:
        question: The metrics investigation question
        graph: Optional pre-built graph
        log_context: Optional context from log agent for correlation analysis

    Note: React agents don't have built-in reflection scores like LATS,
    so confidence is set to a default value.
    """
    if log_context:
        enriched_question = f"{question}\n\nLog Analysis Context:\n{log_context}"
    else:
        enriched_question = question

    graph = graph or build_graph()

    result = graph.invoke({"messages": [HumanMessage(content=enriched_question)]})

    messages = result.get("messages", [])
    final_message = messages[-1] if messages else None
    summary = getattr(final_message, "content", "No solution found")

    evidence = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                if tool_name == "compare_scenarios_metrics":
                    args = tool_call.get("args", {})
                    evidence.append(
                        f"Compared scenarios: {args.get('error_scenario', 'unknown')}"
                    )
                elif tool_name == "generate_comparison_charts":
                    evidence.append("Generated comparison charts")
                elif tool_name == "load_metrics_csv":
                    evidence.append("Loaded metrics data")

    confidence = 0.8

    return AgentResult(
        agent_name="react_metrics_agent",
        summary=summary,
        confidence=confidence,
        evidence=evidence[:10],
        reflections=[], 
        needs_collaboration=False,  
        suggested_next_agent=None,
        confidence_breakdown={},
    )


def default_question(settings: Optional[Settings] = None) -> str:
    """Generate the default question for metrics analysis."""
    settings = settings or get_settings()

    test_scenario = settings.dataset.logs_root.name

    return (
        f"Analyze the metrics for the test case '{test_scenario}' from the CSV "
        f"'{settings.dataset.metrics_csv}'. Compare it against the baseline 'correct' scenario and "
        f"identify which metrics show significant anomalies that could explain the authentication errors. "
        f"Generate comparison charts to '{settings.dataset.metrics_chart_dir}'."
    )


__all__ = ["build_graph", "run", "DEFAULT_METRICS_QUESTION", "default_question"]