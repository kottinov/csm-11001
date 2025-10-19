from __future__ import annotations

from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..config.settings import Settings, get_settings
from ..core.lats import LATSAgentSpec, build_agent_graph
from ..core.result import AgentResult
from ..tools.filesystem import LOG_TOOLS

DEFAULT_LOG_QUESTION = (
    "Analyze the authentication errors in the configured log dataset and determine the root cause."
)


def _build_initial_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Root Cause Analysis (RCA) expert AI assistant. "
                "Investigate system issues by analyzing log files and related data.\n\n"
                "Toolbox:\n"
                "- list_files: Explore directory structures\n"
                "- read_file: Read complete log files\n"
                "- grep_file: Search for keywords in specific files\n"
                "- search_directory: Search for keywords across files recursively\n\n"
                "Dataset location:\n"
                "- The primary log dataset for this investigation is located at: {logs_root}\n\n"
                "Investigation guidelines:\n"
                "1. Inspect the directory structure to orient yourself\n"
                "2. Search for frequent error keywords (error, exception, failed, 401, 500, etc.)\n"
                "3. Use grep_file or search_directory to find specific errors (NOT read_file for large logs)\n"
                "4. Read small samples with read_file only after identifying relevant files via grep\n"
                "5. Correlate findings to identify patterns\n"
                "6. Propose the root cause backed by evidence\n\n"
                "IMPORTANT: Log files can be very large. Always use grep_file/search_directory first.\n"
                "Only use read_file on small files or after you've identified the specific file to examine.\n"
                "Always base conclusions on tool evidence.",
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
                "You are evaluating a Root Cause Analysis response. "
                "Reflect on the candidate answer with focus on:\n"
                "1. Appropriate tool usage and log investigation\n"
                "2. Depth and accuracy of the analysis\n"
                "3. Clarity in identifying the root cause\n"
                "4. Evidence-backed reasoning",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="candidate"),
        ]
    )


def build_graph(settings: Optional[Settings] = None):
    """Create the compiled LangGraph LATS graph for log analysis."""
    settings = settings or get_settings()
    llm = ChatAnthropic(model=settings.anthropic.log_model)
    initial_prompt = _build_initial_prompt().partial(
        logs_root=str(settings.dataset.logs_root)
    )
    spec = LATSAgentSpec(
        name="log_agent",
        llm=llm,
        tools=LOG_TOOLS,
        initial_prompt=initial_prompt,
        reflection_prompt=_build_reflection_prompt(),
        max_height=settings.runtime.log_agent_max_depth,
        candidate_batch_size=settings.runtime.candidate_batch_size,
    )
    return build_agent_graph(spec)


def _extract_evidence_from_trajectory(trajectory: List) -> List[str]:
    """Extract file paths and evidence references from tool calls in trajectory."""
    evidence = []
    for msg in trajectory:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            for tool_call in msg.tool_calls:
                if tool_call.get("name") in ["read_file", "grep_file"]:
                    args = tool_call.get("args", {})
                    if "file_path" in args:
                        evidence.append(f"Analyzed: {args['file_path']}")
                elif tool_call.get("name") == "search_directory":
                    args = tool_call.get("args", {})
                    if "keyword" in args:
                        evidence.append(f"Searched for: {args['keyword']}")
        elif isinstance(msg, ToolMessage):
            content = str(msg.content)
            if content and not content.startswith("Error"):
                lines = content.split("\n")
                if len(lines) > 3:
                    evidence.append(f"Found {len(lines)} results")
    return evidence[:10]

def run(question: str = DEFAULT_LOG_QUESTION, graph=None) -> AgentResult:
    """Execute the log analysis agent and return structured result with reflection scores.

    This enables reflection-based coordination: the agent assesses its own
    confidence and can request metrics analysis if needed.
    """
    graph = graph or build_graph()
    result = graph.invoke({"input": question})
    solution_node = result["root"].best_solution()
    trajectory = solution_node.get_trajectory(include_reflections=False)

    final_message = trajectory[-1] if trajectory else None
    summary = getattr(final_message, "content", "No solution found")

    reflection = solution_node.reflection
    confidence = reflection.normalized_score

    evidence = _extract_evidence_from_trajectory(trajectory)

    needs_collaboration = confidence < 0.7
    suggested_next = "metrics_agent" if needs_collaboration else None

    return AgentResult(
        agent_name="log_agent",
        summary=summary,
        confidence=confidence,
        evidence=evidence,
        reflections=reflection.reflections,
        needs_collaboration=needs_collaboration,
        suggested_next_agent=suggested_next,
        confidence_breakdown=reflection.confidence_breakdown,
    )


def run_legacy(question: str = DEFAULT_LOG_QUESTION, graph=None) -> str:
    """Legacy interface: Execute the log analysis agent and return the final solution text.

    Deprecated: Use run() instead for structured results with coordination support.
    """
    result = run(question, graph)
    return result.summary


def default_question(settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    return (
        f"Analyze the logs in '{settings.dataset.logs_root}' and determine the root cause "
        "of the 401 authentication errors."
    )


__all__ = ["build_graph", "run", "DEFAULT_LOG_QUESTION", "default_question"]
