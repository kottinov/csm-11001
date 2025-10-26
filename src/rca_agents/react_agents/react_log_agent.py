"""React agent for log analysis - baseline implementation."""
from __future__ import annotations

from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from ..config.settings import Settings, get_settings
from ..core.result import AgentResult
from ..tools.filesystem import LOG_TOOLS

DEFAULT_LOG_QUESTION = (
    "Analyze the authentication errors in the configured log dataset and determine the root cause."
)


def _build_system_prompt(settings: Settings) -> str:
    """Build the system prompt for the React log agent."""
    return (
        "You are a Root Cause Analysis (RCA) expert AI assistant. "
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
        "IMPORTANT: Log files can be very large. Always use grep_file/search_directory first.\n"
        "Only use read_file on small files or after you've identified the specific file to examine.\n"
        "Always base conclusions on tool evidence."
    )


def build_graph(settings: Optional[Settings] = None):
    """Create the React agent graph for log analysis.

    This is a baseline implementation using LangGraph's create_react_agent
    for comparison with the LATS-based approach.
    """
    settings = settings or get_settings()
    llm = ChatAnthropic(model=settings.anthropic.log_model)

    system_prompt = _build_system_prompt(settings)

    graph = create_react_agent(
        llm,
        tools=LOG_TOOLS,
        prompt=system_prompt,
    )

    return graph


def run(question: str = DEFAULT_LOG_QUESTION, graph=None) -> AgentResult:
    """Execute the React log analysis agent and return structured result.

    Note: React agents don't have built-in reflection scores like LATS,
    so confidence is set to a default value.
    """
    graph = graph or build_graph()

    result = graph.invoke({"messages": [HumanMessage(content=question)]})

    messages = result.get("messages", [])
    final_message = messages[-1] if messages else None
    summary = getattr(final_message, "content", "No solution found")

    evidence = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                if tool_name in ["read_file", "grep_file"]:
                    args = tool_call.get("args", {})
                    if "file_path" in args:
                        evidence.append(f"Analyzed: {args['file_path']}")
                elif tool_name == "search_directory":
                    args = tool_call.get("args", {})
                    if "keyword" in args:
                        evidence.append(f"Searched for: {args['keyword']}")

    confidence = 0.8

    return AgentResult(
        agent_name="react_log_agent",
        summary=summary,
        confidence=confidence,
        evidence=evidence[:10],
        reflections=[],  
        needs_collaboration=False,  
        suggested_next_agent=None,
        confidence_breakdown={},
    )


def default_question(settings: Optional[Settings] = None) -> str:
    """Generate the default question for log analysis."""
    settings = settings or get_settings()
    return (
        f"Analyze the logs in '{settings.dataset.logs_root}' and determine the root cause "
        "of the 401 authentication errors."
    )


__all__ = ["build_graph", "run", "DEFAULT_LOG_QUESTION", "default_question"]