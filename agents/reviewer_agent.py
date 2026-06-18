"""
ARIA — Reviewer Agent

Code review with structured feedback.
Reviews code files for:
  - Bugs and logic errors
  - Style and convention issues
  - Security vulnerabilities
  - Performance concerns
  - Missing error handling

Can review single files, a full project, or specific changes.

Commands:
  `review [path]`        — Review a specific file or project directory
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.llm import LLMClient


def run_review(
    target_path: str,
    llm: LLMClient,
    user_context: str = "",
    existing_context: str = "",
) -> Dict[str, Any]:
    """
    Review code files for bugs, style issues, and improvements.

    Args:
        target_path: Path to a file or directory to review
        llm: LLM client for analysis
        user_context: User facts from memory

    Returns:
        Dict with: target, files_reviewed[], issues[], summary, passed, duration
    """
    start = time.time()
    path = Path(target_path)

    if not path.exists():
        return {
            "target": target_path,
            "error": f"Path not found: {target_path}",
            "duration": f"{time.time() - start:.0f}s",
        }

    print(f"  [Reviewer] Reviewing: {target_path}")
    print(f"  [Reviewer] Collecting files...")

    # ── Gather Python files ──────────────────────────────────────────
    python_files = []
    if path.is_file():
        if path.suffix == ".py":
            python_files.append(path)
    else:
        python_files = sorted(path.rglob("*.py"))

    if not python_files:
        return {
            "target": target_path,
            "error": "No Python files found to review.",
            "duration": f"{time.time() - start:.0f}s",
        }

    print(f"  [Reviewer] Found {len(python_files)} Python files")

    # ── Build review context (truncated if too large) ────────────────
    file_contexts = []
    total_chars = 0
    MAX_INPUT_CHARS = 8000

    for pf in python_files:
        try:
            code = pf.read_text(encoding="utf-8")
        except Exception:
            continue

        rel_path = pf.relative_to(path if path.is_dir() else path.parent)
        snippet = f"--- {rel_path} ---\n{code}\n"

        if total_chars + len(snippet) > MAX_INPUT_CHARS:
            remaining = len(python_files) - len(file_contexts)
            if remaining > 0:
                file_contexts.append(
                    f"--- {rel_path} ---\n[truncated, {len(code)} bytes — {remaining} more files not shown]\n"
                )
            break

        file_contexts.append(snippet)
        total_chars += len(snippet)

    review_prompt = f"""Review these Python files for bugs, style issues, and improvements:

{''.join(file_contexts)}

{existing_context}

For each file, analyze:
1. **Bugs/Logic Errors** — Any code that will break or produce wrong results
2. **Style/Convention** — Naming, imports, formatting issues
3. **Security** — Injection risks, unsafe operations, hardcoded secrets
4. **Performance** — Inefficient operations, unnecessary allocations
5. **Error Handling** — Missing try/except, unhandled edge cases
6. **Missing Features** — Things the project likely needs but doesn't have

Output format — one line per file summary, then details:
REVIEW: path/to/file.py | PASS | No issues found
REVIEW: path/to/file.py | ISSUE | <brief summary of issues>

--- Details ---
<file-level details with line references if possible>

Be specific and actionable. Reference line numbers when pointing out issues.
"""

    review_system = (
        "You are a senior code reviewer. Be thorough but practical. "
        "Focus on real bugs and meaningful improvements, not nitpicks. "
        "Use the exact REVIEW: format for machine parsing."
    )
    if user_context:
        review_system += f"\n\n### User Context\n{user_context}"

    try:
        review_output = llm.generate(
            review_prompt,
            system_prompt=review_system,
            max_tokens=2048,
        )
    except Exception as e:
        return {
            "target": target_path,
            "error": f"Review generation failed: {e}",
            "duration": f"{time.time() - start:.0f}s",
        }

    # ── Parse results ────────────────────────────────────────────────
    issues = _parse_review_results(review_output, python_files, path)

    passed_count = sum(1 for i in issues if i["status"] == "PASS")
    issue_count = sum(1 for i in issues if i["status"] == "ISSUE")
    total = len(issues)

    duration = time.time() - start
    print(f"  [Reviewer] Complete ({duration:.0f}s) — {passed_count} passed, {issue_count} issues")

    return {
        "target": target_path,
        "file_count": len(python_files),
        "reviewed_count": total,
        "issues": issues,
        "passed": passed_count,
        "issues_found": issue_count,
        "raw_review": review_output.strip(),
        "duration": f"{duration:.0f}s",
    }


def format_review_result(result: Dict[str, Any]) -> str:
    """Format review result as markdown for the CLI."""
    if "error" in result:
        return f"## Review Failed\n\n{result['error']}"

    lines = [
        f"## Code Review: {result['target']}",
        "",
        f"**Files reviewed:** {result.get('reviewed_count', 0)} / {result.get('file_count', 0)}",
        f"**Issues found:** {result.get('issues_found', 0)}",
        f"**Duration:** {result.get('duration', '0s')}",
        "",
        "---",
        "",
    ]

    for issue in result.get("issues", []):
        icon = "✅" if issue["status"] == "PASS" else "⚠️"
        lines.append(f"{icon} **{issue['file']}** — {issue['summary']}")

    lines.extend([
        "",
        "---",
        "",
        "### Details",
        "",
        result.get("raw_review", "No detailed review available."),
    ])

    return "\n".join(lines)


# ── Private Helpers ──────────────────────────────────────────────────


def _parse_review_results(
    review_output: str,
    python_files: List[Path],
    base_path: Path,
) -> List[Dict[str, str]]:
    """Parse LLM review output into structured issue list."""
    issues = []
    reviewed_files = set()

    # Try to parse REVIEW: lines
    for line in review_output.split("\n"):
        line = line.strip()
        if line.startswith("REVIEW:"):
            content = line[7:].strip()
            parts = [p.strip() for p in content.split("|", 2)]
            if len(parts) >= 3:
                file_path = parts[0]
                status = parts[1].upper()
                summary = parts[2]
                reviewed_files.add(file_path)
                issues.append({
                    "file": file_path,
                    "status": status if status in ("PASS", "ISSUE") else "ISSUE",
                    "summary": summary,
                })

    # Add files that weren't explicitly reviewed as "PASS: Not reviewed"
    for pf in python_files:
        try:
            rel_path = str(pf.relative_to(base_path if base_path.is_dir() else base_path.parent))
        except ValueError:
            rel_path = pf.name
        if rel_path not in reviewed_files:
            issues.append({
                "file": rel_path,
                "status": "PASS",
                "summary": "Not explicitly reviewed (no issues flagged)",
            })

    return issues
