"""Tools for generating and compiling LaTeX incident reports."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    path = Path(file_path).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"✓ Wrote {len(content)} characters to {path}"
    except OSError as exc:
        return f"Error writing to '{path}': {exc}"


@tool
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in file (first occurrence only)."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return f"Error: file '{path}' does not exist"
    if not path.is_file():
        return f"Error: '{path}' is not a regular file"

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Error reading '{path}': {exc}"

    if old_string not in content:
        return f"Error: old_string not found in {path}"

    updated = content.replace(old_string, new_string, 1)

    try:
        path.write_text(updated, encoding="utf-8")
        return f"✓ Edited {path} (replaced {len(old_string)} → {len(new_string)} characters)"
    except OSError as exc:
        return f"Error writing to '{path}': {exc}"


@tool
def compile_latex(tex_path: str) -> str:
    """Compile a LaTeX file to PDF using pdflatex.

    Returns the path to the PDF on success, or compilation errors on failure.
    """
    if not shutil.which("pdflatex"):
        return "Error: pdflatex not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)"

    path = Path(tex_path).expanduser().resolve()

    if not path.exists():
        return f"Error: LaTeX file '{path}' does not exist"
    if not path.suffix == ".tex":
        return f"Error: '{path}' is not a .tex file"

    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(path)],
            capture_output=True,
            text=True,
            cwd=path.parent,
            timeout=60,
        )

        pdf_path = path.with_suffix(".pdf")

        if result.returncode == 0 and pdf_path.exists():
            return f"✓ Successfully compiled to: {pdf_path}"
        else:
            output_lines = result.stdout.splitlines()
            errors = [line for line in output_lines if line.startswith("!")]
            error_summary = "\n".join(errors[:10]) if errors else result.stdout[-500:]
            return f"✗ LaTeX compilation failed:\n{error_summary}"

    except subprocess.TimeoutExpired:
        return "Error: LaTeX compilation timed out after 60 seconds"
    except Exception as exc:
        return f"Error during compilation: {exc}"


REPORT_TOOLS = [write_file, edit_file, compile_latex]

__all__ = ["write_file", "edit_file", "compile_latex", "REPORT_TOOLS"]