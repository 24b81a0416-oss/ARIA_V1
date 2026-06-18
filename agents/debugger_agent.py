"""
ARIA — Debugger Agent

Bug fixing and code repair.
Takes a bug report (from the reviewer agent or user) and generates fixes.
Can:
  - Fix bugs identified by the reviewer agent
  - Fix bugs described by the user
  - Apply fixes to the files

Commands:
  `debug [path]`         — Debug and fix issues in a file or project
  `debug [description]`  — Debug a specific issue described by the user
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.llm import LLMClient
from utils.file_parser import parse_files


def run_debug(
    target_path: str,
    llm: LLMClient,
    issue_description: str = "",
    user_context: str = "",
    existing_context: str = "",
    apply_fixes: bool = False,
) -> Dict[str, Any]:
    """
    Debug code issues and generate fixes.

    Args:
        target_path: Path to file or directory to debug
        llm: LLM client for analysis
        issue_description: Specific issue to fix (from reviewer or user)
        user_context: User facts from memory
        existing_context: Previous analysis context
        apply_fixes: If True, write fixes to disk. If False, only analyze and report.

    Returns:
        Dict with: target, files_analyzed[], fixes[], summary, duration
    """
    start = time.time()
    path = Path(target_path)

    if not path.exists():
        return {
            "target": target_path,
            "error": f"Path not found: {target_path}",
            "duration": f"{time.time() - start:.0f}s",
        }

    print(f"  [Debugger] Analyzing: {target_path}")

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
            "error": "No Python files found to debug.",
            "duration": f"{time.time() - start:.0f}s",
        }

    print(f"  [Debugger] Found {len(python_files)} Python files")

    # ── Build debug context ──────────────────────────────────────────
    file_contexts = []
    total_chars = 0
    MAX_INPUT_CHARS = 10000

    for pf in python_files:
        try:
            code = pf.read_text(encoding="utf-8")
        except Exception:
            continue

        rel_path = pf.relative_to(path if path.is_dir() else path.parent)
        snippet = f"--- {rel_path} ---\n{code}\n"

        if total_chars + len(snippet) > MAX_INPUT_CHARS:
            break

        file_contexts.append(snippet)
        total_chars += len(snippet)

    debug_prompt = f"""Debug and fix the following code:

{''.join(file_contexts)}

{existing_context}

Issue: {issue_description if issue_description else 'Find and fix any bugs, errors, or issues in this code.'}

## Instructions
1. Analyze the code thoroughly for bugs and issues
2. For each issue found, generate the fix
3. Output fixes using the exact markers:

---FILE: path/to/file.py | Brief description of the fix
<fixed code content — the ENTIRE file with the fix applied>
---END FILE

## Focus Areas
- Logic errors and incorrect behavior
- Import errors and missing dependencies
- Type errors and incorrect API usage
- Resource leaks (files, connections not closed)
- Race conditions or async issues
- Missing error handling
- Off-by-one errors and boundary conditions

Output ONLY the files that need fixing. If no bugs found, respond with: NO BUGS FOUND
"""

    debug_system = (
        "You are a senior debugging engineer. Find bugs methodically and "
        "provide complete fixed files. Use the exact file markers specified."
    )
    if user_context:
        debug_system += f"\n\n### User Context\n{user_context}"

    try:
        debug_output = llm.generate(
            debug_prompt,
            system_prompt=debug_system,
            max_tokens=4096,
        )
    except Exception as e:
        return {
            "target": target_path,
            "error": f"Debug analysis failed: {e}",
            "duration": f"{time.time() - start:.0f}s",
        }

    # ── Check if no bugs found ──────────────────────────────────────
    if "NO BUGS FOUND" in debug_output.upper():
        duration = time.time() - start
        print(f"  [Debugger] Complete ({duration:.0f}s) — no bugs found")
        return {
            "target": target_path,
            "bug_count": 0,
            "fixes": [],
            "summary": "No bugs found in the reviewed code.",
            "duration": f"{duration:.0f}s",
        }

    # ── Parse fixes from output ──────────────────────────────────────
    fixes = parse_files(debug_output)

    if not fixes:
        # No parseable fixes — the LLM found issues but didn't format them correctly
        print(f"  [Debugger] Issues identified but couldn't parse fixes from output")
        return {
            "target": target_path,
            "bug_count": 1,
            "fixes": [],
            "summary": "Issues may exist but couldn't parse structured fixes. Check raw output.",
            "raw_output": debug_output.strip(),
            "duration": f"{time.time() - start:.0f}s",
        }

    # ── Apply fixes (only if apply_fixes=True) ───────────────────────
    applied = []
    if apply_fixes:
        for fix in fixes:
            file_path = path.parent / fix["path"] if path.is_file() else path / fix["path"]
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(fix["code"], encoding="utf-8")
                applied.append({
                    "path": fix["path"],
                    "description": fix.get("description", "Bug fix applied"),
                    "size": len(fix["code"]),
                })
                print(f"    ✓ Fixed: {fix['path']} — {fix.get('description', 'bug fix')}")
            except Exception as e:
                print(f"    ✗ Failed to write {fix['path']}: {e}")
    else:
        # Just report what would be fixed
        applied = [{
            "path": fix["path"],
            "description": fix.get("description", "Proposed fix"),
            "size": len(fix["code"]),
        } for fix in fixes]
        for fix in fixes:
            print(f"    → Proposed fix: {fix['path']} — {fix.get('description', 'bug fix')}")

    duration = time.time() - start
    status = "applied" if apply_fixes else "proposed"
    print(f"  [Debugger] Complete ({duration:.0f}s) — {len(fixes)} fixes {status}")

    return {
        "target": target_path,
        "bug_count": len(fixes),
        "fixes_applied": len(applied),
        "fixes": applied,
        "fixes_applied_flag": apply_fixes,
        "summary": f"Found {len(fixes)} issues. " + (
            f"Applied {len(applied)} fixes." if apply_fixes
            else "Review proposed fixes and re-run with `debug` + confirmation to apply."
        ),
        "duration": f"{duration:.0f}s",
    }


def format_debug_result(result: Dict[str, Any]) -> str:
    """Format debug result as markdown for the CLI."""
    if "error" in result:
        return f"## Debug Failed\n\n{result['error']}"

    lines = [
        f"## Debug: {result['target']}",
        "",
        f"**Bugs found:** {result.get('bug_count', 0)}",
        f"**Fixes applied:** {result.get('fixes_applied', 0)}",
        f"**Duration:** {result.get('duration', '0s')}",
        "",
        "---",
        "",
    ]

    if result.get("bug_count", 0) == 0:
        lines.append("✅ No bugs found in the reviewed code.")
    else:
        lines.append("### Fixes Applied")
        lines.append("")
        for fix in result.get("fixes", []):
            lines.append(f"- ✅ `{fix['path']}` — {fix.get('description', 'Fixed')}")

        if result.get("fixes_applied", 0) < result.get("bug_count", 0):
            lines.append("")
            lines.append("⚠️ Some fixes could not be applied automatically.")

    raw = result.get("raw_output")
    if raw:
        lines.extend(["", "---", "", "### Raw Analysis", "", raw])

    return "\n".join(lines)


