from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from langchain_core.tools import tool


def _format_dir_entry(path: Path) -> str:
    kind = "DIR" if path.is_dir() else "FILE"
    return f"{kind}: {path.name}"


@tool
def list_files(directory_path: str) -> str:
    """List files and subdirectories for a given path."""
    directory = Path(directory_path).expanduser().resolve()
    if not directory.exists():
        return f"Error: directory '{directory}' does not exist"
    if not directory.is_dir():
        return f"Error: '{directory}' is not a directory"

    entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    if not entries:
        return "Empty directory"
    return "\n".join(_format_dir_entry(path) for path in entries)


@tool
def read_file(file_path: str, max_lines: int = 500) -> str:
    """Read file contents with utf-8 fallback and error handling.

    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to return (default: 500)
                   Set to -1 to read entire file (use with caution)

    Returns first max_lines of the file to avoid token limit issues.
    For large log files, use grep_file or search_directory instead.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return f"Error: file '{path}' does not exist"
    if not path.is_file():
        return f"Error: '{path}' is not a regular file"

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()

        if max_lines > 0 and len(lines) > max_lines:
            truncated_lines = lines[:max_lines]
            return "\n".join(truncated_lines) + f"\n\n[Truncated: showing first {max_lines} of {len(lines)} lines. Use grep_file to search for specific patterns.]"

        return content
    except OSError as exc:
        return f"Error reading '{path}': {exc}"


@tool
def grep_file(file_path: str, keyword: str) -> str:
    """Return matching lines for a keyword in a file."""
    contents = read_file(file_path)
    if contents.startswith("Error:"):
        return contents

    keyword_lower = keyword.lower()
    matches = []
    for index, line in enumerate(contents.splitlines(), start=1):
        if keyword_lower in line.lower():
            matches.append(f"Line {index}: {line}")

    return "\n".join(matches) if matches else f"No matches for '{keyword}'"


def _iter_candidate_files(directory: Path) -> Iterable[Path]:
    for root, _, files in os.walk(directory):
        for file_name in files:
            yield Path(root) / file_name


@tool
def search_directory(directory_path: str, keyword: str) -> str:
    """Search for a keyword across files in a directory tree."""
    directory = Path(directory_path).expanduser().resolve()
    if not directory.exists():
        return f"Error: directory '{directory}' does not exist"
    if not directory.is_dir():
        return f"Error: '{directory}' is not a directory"

    keyword_lower = keyword.lower()
    results = []
    for file_path in _iter_candidate_files(directory):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        matching_lines = []
        for index, line in enumerate(text.splitlines(), start=1):
            if keyword_lower in line.lower():
                matching_lines.append(f"  Line {index}: {line}")

        if matching_lines:
            results.append(f"\n{file_path}:")
            results.extend(matching_lines[:10])
            if len(matching_lines) > 10:
                remaining = len(matching_lines) - 10
                results.append(f"  ... ({remaining} more matches)")

    return "\n".join(results) if results else f"No matches for '{keyword}'"


LOG_TOOLS = [list_files, read_file, grep_file, search_directory]

__all__ = ["list_files", "read_file", "grep_file", "search_directory", "LOG_TOOLS"]
