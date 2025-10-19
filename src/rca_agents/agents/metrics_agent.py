from __future__ import annotations

from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..config.settings import Settings, get_settings
from ..core.lats import LATSAgentSpec, build_agent_graph
from ..core.result import AgentResult
from ..tools.metrics import METRICS_TOOLS

DEFAULT_METRICS_QUESTION = (
    "Analyze the metrics captured in the configured dataset, compare the error scenario with the baseline, "
    "and identify anomalies that could explain the authentication errors. Generate comparison charts."
)


def _build_initial_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a metrics analysis expert supporting Root Cause Analysis (RCA). "
                "Investigate performance metrics data to understand incidents.\n\n"
                "Available tools:\n"
                "- load_metrics_csv: Inspect scenarios and counts\n"
                "- get_metrics_summary: Summarize metrics for a scenario\n"
                "- compare_scenarios_metrics: Contrast error vs baseline metrics\n"
                "- generate_comparison_charts: Visualize differences\n\n"
                "Dataset location:\n"
                "- The metrics CSV for this investigation is located at: {metrics_csv}\n\n"
                "Guidelines:\n"
                "1. Load the CSV to understand scenarios and sample sizes\n"
                "2. Compare the error scenario with the baseline 'correct'\n"
                "3. Assess goroutines, memory, GC, and system load metrics\n"
                "4. Highlight statistically meaningful differences; acknowledge normal findings\n"
                "5. Generate charts when requested, noting output locations\n"
                "6. Report findings honestly, especially when metrics look nominal.",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )


def _build_reflection_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are grading a metrics analysis candidate response. Consider:\n"
                "1. Tool usage and coverage of metrics scenarios\n"
                "2. Accuracy when describing anomalies or normalcy\n"
                "3. Correct treatment of sample sizes and statistical confidence\n"
                "4. Clear linkage between data and proposed root causes\n"
                "5. Avoidance of unfounded performance claims",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="candidate"),
        ]
    )


def build_graph(settings: Optional[Settings] = None):
    """Create the compiled LangGraph LATS graph for metrics analysis."""
    settings = settings or get_settings()
    llm = ChatAnthropic(model=settings.anthropic.metrics_model)
    initial_prompt = _build_initial_prompt().partial(
        metrics_csv=str(settings.dataset.metrics_csv)
    )
    spec = LATSAgentSpec(
        name="metrics_agent",
        llm=llm,
        tools=METRICS_TOOLS,
        initial_prompt=initial_prompt,
        reflection_prompt=_build_reflection_prompt(),
        max_height=settings.runtime.metrics_agent_max_depth,
        candidate_batch_size=settings.runtime.candidate_batch_size,
    )
    return build_agent_graph(spec)


def _extract_metrics_evidence_from_trajectory(trajectory: List) -> List[str]:
    """Extract metrics analysis evidence from tool calls in trajectory."""
    evidence = []
    for msg in trajectory:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            for tool_call in msg.tool_calls:
                if tool_call.get("name") == "compare_scenarios_metrics":
                    args = tool_call.get("args", {})
                    evidence.append(f"Compared scenarios: {args.get('error_scenario', 'unknown')}")
                elif tool_call.get("name") == "generate_comparison_charts":
                    evidence.append("Generated comparison charts")
                elif tool_call.get("name") == "load_metrics_csv":
                    evidence.append("Loaded metrics data")
        elif isinstance(msg, ToolMessage):
            content = str(msg.content)
            if "goroutines" in content.lower() or "memory" in content.lower():
                evidence.append("Analyzed system metrics")
    return evidence[:10] 


def run(question: str = DEFAULT_METRICS_QUESTION, graph=None, log_context: Optional[str] = None) -> AgentResult:
    """Execute the metrics analysis agent and return structured result with reflection scores.

    Args:
        question: The metrics investigation question
        graph: Optional pre-built graph
        log_context: Optional context from log agent for correlation analysis

    This enables reflection-based coordination: the agent can assess correlation
    with log findings and determine if the metrics confirm or contradict log analysis.
    """
    if log_context:
        enriched_question = f"{question}\n\nLog Analysis Context:\n{log_context}"
    else:
        enriched_question = question

    graph = graph or build_graph()
    result = graph.invoke({"input": enriched_question})
    solution_node = result["root"].best_solution()
    trajectory = solution_node.get_trajectory(include_reflections=False)

    final_message = trajectory[-1] if trajectory else None
    summary = getattr(final_message, "content", "No solution found")

    reflection = solution_node.reflection
    confidence = reflection.normalized_score

    evidence = _extract_metrics_evidence_from_trajectory(trajectory)

    needs_collaboration = confidence < 0.5

    return AgentResult(
        agent_name="metrics_agent",
        summary=summary,
        confidence=confidence,
        evidence=evidence,
        reflections=reflection.reflections,
        needs_collaboration=needs_collaboration,
        suggested_next_agent=None,  # delegate to report gen
        confidence_breakdown=reflection.confidence_breakdown,
    )


def run_legacy(question: str = DEFAULT_METRICS_QUESTION, graph=None) -> str:
    """Legacy interface: Execute the metrics analysis agent and return the final solution text.

    Deprecated: Use run() instead for structured results with coordination support.
    """
    result = run(question, graph)
    return result.summary


def default_question(settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()

    test_scenario = settings.dataset.logs_root.name

    return (
        f"Analyze the metrics for the test case '{test_scenario}' from the CSV "
        f"'{settings.dataset.metrics_csv}'. Compare it against the baseline 'correct' scenario and "
        f"identify which metrics show significant anomalies that could explain the authentication errors. "
        f"Generate comparison charts to '{settings.dataset.metrics_chart_dir}'."
    )


__all__ = ["build_graph", "run", "DEFAULT_METRICS_QUESTION", "default_question"]
