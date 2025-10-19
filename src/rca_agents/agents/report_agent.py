"""Report generation agent using ReAct pattern with file editing tools."""
from __future__ import annotations

from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from ..config.settings import Settings, get_settings
from ..core.result import AgentResult
from ..tools.filesystem import list_files, read_file
from ..tools.report import REPORT_TOOLS


def _truncate_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "... [truncated]"


def build_report_agent(settings: Optional[Settings] = None):
    """Create a ReAct agent for generating LaTeX incident reports.

    The agent has access to:
    - list_files: Find generated charts
    - read_file: Review its own draft
    - write_file: Create initial report.tex
    - edit_file: Fix mistakes via old_string/new_string replacement
    - compile_latex: Generate PDF and check for errors
    """
    settings = settings or get_settings()
    llm = ChatAnthropic(
        model=settings.anthropic.supervisor_model,
        max_tokens_to_sample=4000,
        temperature=0,
    )

    system_prompt = """You are a technical report writer specializing in incident analysis.
    Your task is to generate professional LaTeX incident reports based on investigation findings.

    Available tools:
    - list_files(directory): Find chart images and other files
    - read_file(path): Read file contents (including your own draft)
    - write_file(path, content): Create a new file
    - edit_file(path, old_string, new_string): Replace text in a file (exact match required)
    - compile_latex(path): Compile .tex to PDF and check for errors

    Workflow:
    1. List the charts directory to find available visualizations
    2. Design and write a complete LaTeX document based on the findings
    3. Write the document in manageable chunks (≤1500 characters per tool call)
    4. Read back the report to review quality and completeness
    5. Edit any issues you find (unclear wording, missing evidence, LaTeX errors)
    6. Compile to PDF - if errors occur, read the error message and fix them
    7. Iterate until compilation succeeds

    Report design is up to you - choose structure and sections that best present the findings.

    Guidelines:
    - Write in academic PhD-style narrative prose throughout the entire document
    - NEVER use bullet points, numbered lists, or itemized formats
    - Use flowing paragraphs with smooth transitions between ideas
    - Employ formal academic language typical of scholarly publications
    - Write in third person or passive voice where appropriate
    - Integrate evidence naturally within narrative sentences (e.g., "Analysis of access.log:127 revealed...")
    - Include charts with \\includegraphics and discuss them in surrounding text
    - Express uncertainty with formal academic hedging language (e.g., "The evidence suggests...", "It appears that...")
    - Weave confidence scores naturally into the narrative
    - Structure sections with clear topic sentences and supporting paragraphs
    - Make it publication-ready for academic journals or technical conferences
    - Aim for 1-2 pages of dense, scholarly prose

    LaTeX tips:
    - Start with \\documentclass, \\usepackage{graphicx}, \\usepackage{hyperref}
    - Use \\includegraphics[width=0.7\\textwidth]{path/to/chart.png}
    - Escape special characters: \\_, \\%, \\&, \\#
    - Use \\texttt{} for log entries and file paths
    - Compile catches syntax errors - read and fix them

    IMPORTANT:
    - Keep each write_file or edit_file call under 1500 characters.
    - When calling write_file, always provide both the file path and the LaTeX content.
    - Use edit_file to update specific sections instead of rewriting the entire document."""

    all_tools = [list_files, read_file] + REPORT_TOOLS
    return create_react_agent(llm, all_tools, prompt=system_prompt)


def run(
    log_result: AgentResult,
    metrics_result: AgentResult,
    output_path: str = "reports/incident_report.tex",
    settings: Optional[Settings] = None,
    recursion_limit: int = 200,
) -> str:
    """Generate an incident report and compile it to PDF.

    Args:
        log_result: Findings from log_agent
        metrics_result: Findings from metrics_agent
        output_path: Where to write the .tex file
        settings: Optional settings override
        recursion_limit: Maximum number of agent graph steps before aborting

    Returns:
        Path to the generated PDF (or error message)
    """
    settings = settings or get_settings()
    agent = build_report_agent(settings)

    log_summary = _truncate_text(log_result.summary, 3500)
    metrics_summary = _truncate_text(metrics_result.summary, 3500)
    log_reflection = _truncate_text(log_result.reflections or "", 1500)
    metrics_reflection = _truncate_text(metrics_result.reflections or "", 1500)

    max_evidence_items = 12
    trimmed_log_evidence = list(log_result.evidence[:max_evidence_items])
    if len(log_result.evidence) > max_evidence_items:
        trimmed_log_evidence.append("... (additional evidence truncated)")
    log_evidence = "\n".join(f"  - {e}" for e in trimmed_log_evidence) or "  - (none)"

    trimmed_metrics_evidence = list(metrics_result.evidence[:max_evidence_items])
    if len(metrics_result.evidence) > max_evidence_items:
        trimmed_metrics_evidence.append("... (additional evidence truncated)")
    metrics_evidence = "\n".join(f"  - {e}" for e in trimmed_metrics_evidence) or "  - (none)"

    task = f"""Generate a professional incident report at: {output_path}

    LOG ANALYSIS FINDINGS:
    Summary: {log_summary}
    Confidence: {log_result.confidence:.2f} / 1.0
    Evidence:
    {log_evidence}
    Reflection: {log_reflection}

    METRICS ANALYSIS FINDINGS:
    Summary: {metrics_summary}
    Confidence: {metrics_result.confidence:.2f} / 1.0
    Evidence:
    {metrics_evidence}
    Reflection: {metrics_reflection}

    COORDINATION METADATA:
    - Log agent needs collaboration: {log_result.needs_collaboration}
    - Metrics agent needs collaboration: {metrics_result.needs_collaboration}
    - Charts directory: {settings.dataset.metrics_chart_dir}

    Instructions:
    1. List charts directory to find available images: {settings.dataset.metrics_chart_dir}
    2. Design and write a complete LaTeX document in academic PhD-style narrative format
       - Use ONLY flowing paragraphs and continuous prose
       - ABSOLUTELY NO bullet points, numbered lists, or itemized formats anywhere
       - Write in formal academic style suitable for scholarly publication
    3. Include relevant charts as figures with narrative discussion
    4. Read back your draft to verify quality and adherence to narrative style
    5. Edit any issues you find, especially removing any lists or bullet points
    6. Compile to PDF with compile_latex('{output_path}')
    7. If compilation fails, fix errors and retry
    8. Respond with "REPORT COMPLETE: <pdf_path>" when done"""

    config = {"recursion_limit": recursion_limit} if recursion_limit else None

    messages = [HumanMessage(content=task)]

    for _ in range(5):
        try:
            result = agent.invoke({"messages": messages}, config=config)
        except GraphRecursionError as exc:
            raise RuntimeError(
                "Report agent exceeded the recursion limit before completing. "
                "Raise the recursion_limit parameter or review tool traces."
            ) from exc

        messages = result["messages"]
        final_message = messages[-1]
        content = getattr(final_message, "content", "")

        if isinstance(content, str) and "REPORT COMPLETE" in content:
            return content

        if isinstance(content, str) and content.strip().lower().startswith("sorry, need more steps"):
            messages.append(
                HumanMessage(
                    content=(
                        "You have not completed the report yet. "
                        "Continue the existing plan, finish writing and editing the LaTeX document, "
                        "compile it successfully, and respond with 'REPORT COMPLETE: <pdf_path>'."
                    )
                )
            )
            continue

        if isinstance(content, str) and content.strip():
            return content

    return getattr(messages[-1], "content", "Report agent did not produce a final response.")


__all__ = ["build_report_agent", "run"]
