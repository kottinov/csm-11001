"""React multi-agent supervisor - baseline implementation using LangGraph supervisor pattern."""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing_extensions import TypedDict

from ..config.settings import Settings, get_settings
from ..core.result import AgentResult
from ..tools.filesystem import LOG_TOOLS
from ..tools.metrics import METRICS_TOOLS


class SupervisorState(MessagesState):
    """State for the supervisor graph."""

    next: str


def _build_log_system_prompt(settings: Settings) -> str:
    """Build the system prompt for the log analysis agent."""
    return (
        "You are a Root Cause Analysis (RCA) expert AI assistant specializing in log analysis. "
        "Investigate system issues by analyzing log files and related data.\n\n"
        "Toolbox:\n"
        "- list_files: Explore directory structures\n"
        "- read_file: Read complete log files\n"
        "- grep_file: Search for keywords in specific files\n"
        "- search_directory: Search for keywords across files recursively\n\n"
        f"Dataset location:\n"
        f"- The primary log dataset for this investigation is located at: {settings.dataset.logs_root}\n\n"
        "Investigation guidelines:\n"
        "1. Inspect the directory structure to orient yourself\n"
        "2. Search for frequent error keywords (error, exception, failed, 401, 500, etc.)\n"
        "3. Use grep_file or search_directory to find specific errors (NOT read_file for large logs)\n"
        "4. Read small samples with read_file only after identifying relevant files via grep\n"
        "5. Correlate findings to identify patterns\n"
        "6. Propose the root cause backed by evidence\n\n"
        "IMPORTANT: Log files can be very large. Always use grep_file/search_directory first."
    )


def _build_metrics_system_prompt(settings: Settings) -> str:
    """Build the system prompt for the metrics analysis agent."""
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


def _create_supervisor_prompt() -> str:
    """Create the supervisor prompt for routing between agents."""
    return (
        "You are a Root Cause Analysis (RCA) supervisor managing a team of specialized agents. "
        "Your team consists of:\n"
        "- log_analyzer: Expert in analyzing log files for errors and patterns\n"
        "- metrics_analyzer: Expert in analyzing performance metrics and system health\n\n"
        "Given the conversation history, decide which agent should act next, or if the analysis is complete.\n"
        "Typically, you should:\n"
        "1. Start with log_analyzer to identify errors in logs\n"
        "2. Then use metrics_analyzer to correlate with system metrics\n"
        "3. Choose FINISH when both analyses are complete\n\n"
        "Available options: log_analyzer, metrics_analyzer, FINISH"
    )


def build_supervisor_graph(settings: Optional[Settings] = None):
    """Build the React multi-agent supervisor graph.

    This creates a supervisor that coordinates between log and metrics analysis agents,
    following the LangGraph agent supervisor pattern.
    """
    settings = settings or get_settings()

    supervisor_llm = ChatAnthropic(model=settings.anthropic.supervisor_model)

    log_llm = ChatAnthropic(model=settings.anthropic.log_model)
    metrics_llm = ChatAnthropic(model=settings.anthropic.metrics_model)

    log_agent = create_react_agent(
        log_llm,
        tools=LOG_TOOLS,
        prompt=_build_log_system_prompt(settings),
    )

    metrics_agent = create_react_agent(
        metrics_llm,
        tools=METRICS_TOOLS,
        prompt=_build_metrics_system_prompt(settings),
    )

    members = ["log_analyzer", "metrics_analyzer"]
    options = members + ["FINISH"]

    def supervisor_node(state: SupervisorState) -> Command[Literal["log_analyzer", "metrics_analyzer", "__end__"]]:
        """Supervisor decides which agent to call next."""
        messages = [
            {"role": "system", "content": _create_supervisor_prompt()},
        ] + state["messages"]

        response = supervisor_llm.invoke(messages)
        content = response.content.strip().lower()

        next_agent = None
        for option in options:
            if option.lower() in content:
                next_agent = option
                break

        if not next_agent or next_agent == "FINISH":
            return Command(goto=END)

        agent_map = {
            "log_analyzer": "log_analyzer",
            "metrics_analyzer": "metrics_analyzer",
        }

        return Command(goto=agent_map.get(next_agent, END))

    # Define agent nodes
    def log_analyzer_node(state: SupervisorState):
        """Log analyzer agent node."""
        result = log_agent.invoke(state)
        return {
            "messages": result["messages"],
        }

    def metrics_analyzer_node(state: SupervisorState):
        """Metrics analyzer agent node."""
        result = metrics_agent.invoke(state)
        return {
            "messages": result["messages"],
        }

    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("log_analyzer", log_analyzer_node)
    workflow.add_node("metrics_analyzer", metrics_analyzer_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_edge("log_analyzer", "supervisor")
    workflow.add_edge("metrics_analyzer", "supervisor")

    return workflow.compile()


def run_supervisor(
    question: str,
    graph=None,
    settings: Optional[Settings] = None,
) -> AgentResult:
    """Execute the React multi-agent supervisor.

    Args:
        question: The investigation question
        graph: Optional pre-built graph
        settings: Optional settings

    Returns:
        AgentResult containing the collaborative analysis
    """
    settings = settings or get_settings()
    graph = graph or build_supervisor_graph(settings)

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"recursion_limit": 20},
    )

    messages = result.get("messages", [])
    final_message = messages[-1] if messages else None
    summary = getattr(final_message, "content", "No solution found")

    evidence = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                args = tool_call.get("args", {})

                if tool_name in ["read_file", "grep_file"] and "file_path" in args:
                    evidence.append(f"Analyzed: {args['file_path']}")
                elif tool_name == "search_directory" and "keyword" in args:
                    evidence.append(f"Searched for: {args['keyword']}")
                elif tool_name == "compare_scenarios_metrics":
                    evidence.append(
                        f"Compared scenarios: {args.get('error_scenario', 'unknown')}"
                    )
                elif tool_name == "generate_comparison_charts":
                    evidence.append("Generated comparison charts")

    confidence = 0.8

    return AgentResult(
        agent_name="react_supervisor",
        summary=summary,
        confidence=confidence,
        evidence=evidence[:10],
        reflections=[],
        needs_collaboration=False,
        suggested_next_agent=None,
        confidence_breakdown={},
    )


def default_supervisor_question(settings: Optional[Settings] = None) -> str:
    """Generate the default question for the supervisor."""
    settings = settings or get_settings()
    return (
        f"Perform a comprehensive Root Cause Analysis of the authentication errors. "
        f"Analyze both the logs in '{settings.dataset.logs_root}' and the metrics in "
        f"'{settings.dataset.metrics_csv}' to identify the root cause of the 401 errors. "
        f"Determine if this is an application-level issue or infrastructure problem."
    )


__all__ = [
    "build_supervisor_graph",
    "run_supervisor",
    "default_supervisor_question",
]