"""
ARIA — Editor Agent

Makes precise edits to existing code files using project-aware context.
Designed to feel like a senior engineer collaborating on your codebase.

Pipeline:
  1. Scan project for context
  2. Plan the changes needed
  3. Make edits using str_replace-like patterns
  4. Show a summary of changes

Commands:
  `edit [description]`        — Edit the current project
  `edit [description] in [path]` — Edit a specific file/directory
"""

from __future__ import annotations

import time
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.llm import LLMClient
from utils.codebase_scanner import scan_project, format_project_context


def plan_edits(
    task: str,
    project_dir: str | Path,
    llm: LLMClient,
    target_path: Optional[str] = None,
    user_context: str = "",
) -> Dict[str, Any]:
    """
    Plan edits without applying them. Returns a plan with diffs.
    Use `apply_edits` to actually write the changes.

    Args:
        task: What to change
        project_dir: Root of the project to edit
        llm: LLM client for planning
        target_path: Optional specific file/directory to scope edits to

    Returns:
        Dict with: task, file_plan[], summary_preview, project_dir
    """
    start = time.time()
    project_dir = Path(project_dir).resolve()
    print(f"  [Editor] Task: {task}")
    print(f"  [Editor] Scanning project: {project_dir.name}...")

    # ── Step 1: Scan project ─────────────────────────────────────────
    scan_result = scan_project(project_dir)
    if "error" in scan_result:
        return {"error": f"Project scan failed: {scan_result['error']}"}

    project_context = format_project_context(scan_result)
    print(f"  [Editor] Found {len(scan_result.get('source_files', []))} source files")

    # ── Step 2: Plan edits ───────────────────────────────────────────
    print(f"  [Editor] Planning changes...")
    plan = _plan_edits(task, project_context, scan_result, llm, target_path)
    if "error" in plan:
        return {"error": plan["error"]}

    files_to_edit = plan.get("files", [])
    if not files_to_edit:
        return {"error": "No files identified for editing", "task": task}

    print(f"  [Editor] Plan: edit {len(files_to_edit)} files")

    # ── Step 3: Generate diffs (preview only) ─────────────────────────
    results = []
    for file_info in files_to_edit:
        file_path = file_info["path"]
        changes = file_info.get("changes", [])

        try:
            preview = _preview_edits(project_dir, file_path, changes)
            results.append(preview)
        except Exception as e:
            results.append({
                "file": file_path,
                "success": False,
                "error": str(e),
            })

    # ── Step 4: Build preview summary ────────────────────────────────
    preview_lines = [
        f"## Edit Plan: {task}",
        "",
        f"**Project:** `{project_dir.name}`",
        f"**Files to edit:** {len(results)}",
        "",
    ]

    for r in results:
        if r.get("error"):
            preview_lines.append(f"❌ `{r['file']}` — {r['error']}")
        else:
            preview_lines.append(f"📝 `{r['file']}")
        if r.get("diff"):
            preview_lines.append(f"   ```diff")
            preview_lines.append(f"   {r['diff'][:600]}")
            preview_lines.append(f"   ```")
            preview_lines.append("")

    preview_lines.append("---")
    preview_lines.append("Type `yes` to apply these changes, or anything else to cancel.")
    preview = "\n".join(preview_lines)

    return {
        "task": task,
        "project_dir": str(project_dir),
        "files": files_to_edit,
        "results": results,
        "preview": preview,
    }


def apply_edits(
    project_dir: str | Path,
    files: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Apply previously planned edits.

    Args:
        project_dir: Root of the project
        files: List of {path, changes} from plan_edits result

    Returns:
        Dict with: results[], summary
    """
    project_dir = Path(project_dir).resolve()
    results = []

    for file_info in files:
        file_path = file_info["path"]
        changes = file_info.get("changes", [])

        try:
            edit_result = _apply_edits(project_dir, file_path, changes)
            results.append(edit_result)
        except Exception as e:
            results.append({
                "file": file_path,
                "success": False,
                "error": str(e),
            })

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    summary_lines = [
        f"**Files edited:** {len(successful)}/{len(results)}",
        "",
    ]
    for r in results:
        icon = "✅" if r.get("success") else "❌"
        summary_lines.append(f"{icon} `{r['file']}`")
        if not r.get("success") and r.get("error"):
            summary_lines.append(f"   _{r['error']}_")

    return {
        "results": results,
        "success_count": len(successful),
        "failed_count": len(failed),
        "summary": "\n".join(summary_lines),
    }


# ── Private: Planning ───────────────────────────────────────────────

def _plan_edits(
    task: str,
    project_context: str,
    scan_result: Dict[str, Any],
    llm: LLMClient,
    target_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Plan what edits to make by analyzing the project context."""
    key_files = scan_result.get("key_files", [])

    # Build a compact list of available files
    file_list = []
    for kf in key_files[:30]:
        path = kf["path"]
        purpose = kf.get("purpose", "")
        size = kf.get("size", 0)
        file_list.append(f"  - {path}  ({size}B, {purpose})")

    file_tree_list = scan_result.get("file_tree", [])
    tree_preview = []
    for entry in file_tree_list[:40]:
        indent = "  " * entry.get("depth", 0)
        icon = "📁" if entry["type"] == "dir" else "📄"
        tree_preview.append(f"{indent}{icon} {entry['name']}")

    plan_prompt = f"""I need to make this change to the project: {task}

## Project Context
{project_context}

## Available Files
{chr(10).join(file_list)}

## Instructions
Identify which files need to be edited and what changes to make.

For EACH file that needs changes, specify:
1. The exact file path (relative to project root)
2. What string to find (the exact existing code to replace)
3. What to replace it with

Output format (one file per block):
```
FILE: path/to/file.py
FIND:
<exact code to find>
REPLACE:
<exact code to replace with>
```

{f'Focus ONLY on files inside: {target_path}' if target_path else ''}
Only list files that actually need changes. Be precise - the FIND text must match exactly.
"""

    plan_system = "You are a senior software engineer. Plan precise edits. Output FIND/REPLACE blocks only."
    if user_context:
        plan_system += f"\n\n### User Context\n{user_context}"

    try:
        plan_response = llm.generate(
            plan_prompt,
            system_prompt=plan_system,
            max_tokens=4096,
        )
    except Exception as e:
        return {"error": f"Edit planning failed: {e}"}

    # Parse the plan into structured edits
    return _parse_edit_plan(plan_response)


def _parse_edit_plan(plan_text: str) -> Dict[str, Any]:
    """Parse LLM edit plan into structured changes per file."""
    files: Dict[str, List[Dict[str, str]]] = {}
    current_file = None
    current_find = []
    current_replace = []
    mode = None  # "find" or "replace"

    for line in plan_text.split("\n"):
        if line.startswith("FILE:"):
            # Save previous file's changes
            if current_file and current_find and current_replace:
                files.setdefault(current_file, []).append({
                    "find": "\n".join(current_find).strip(),
                    "replace": "\n".join(current_replace).strip(),
                })
            current_file = line[5:].strip()
            current_find = []
            current_replace = []
            mode = None
        elif line.strip() == "FIND:":
            mode = "find"
        elif line.strip() == "REPLACE:":
            mode = "replace"
        elif mode == "find":
            current_find.append(line)
        elif mode == "replace":
            current_replace.append(line)

    # Save last file
    if current_file and current_find and current_replace:
        files.setdefault(current_file, []).append({
            "find": "\n".join(current_find).strip(),
            "replace": "\n".join(current_replace).strip(),
        })

    return {
        "files": [
            {"path": path, "changes": changes}
            for path, changes in files.items()
        ]
    }


# ── Private: Previewing Edits (no write) ───────────────────────────

def _preview_edits(
    project_dir: Path,
    file_path: str,
    changes: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Preview edits without writing to disk."""
    full_path = project_dir / file_path
    if not full_path.exists():
        return {"file": file_path, "success": False,
                "error": f"File not found: {file_path}"}

    original = full_path.read_text(encoding="utf-8")
    content = original

    for i, change in enumerate(changes):
        find_text = change.get("find", "")
        replace_text = change.get("replace", "")

        if not find_text:
            continue

        if find_text not in content:
            return {
                "file": file_path,
                "success": False,
                "error": f"Change {i + 1}: Could not find matching text in file",
            }

        content = content.replace(find_text, replace_text, 1)

    # Generate diff
    diff = "\n".join(difflib.unified_diff(
        original.splitlines(),
        content.splitlines(),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    ))

    return {
        "file": file_path,
        "success": True,
        "changes_previewed": len(changes),
        "diff": diff[:1000],
        "size_before": len(original),
        "size_after": len(content),
    }


# ── Private: Applying Edits (writes to disk) ───────────────────────────

def _apply_edits(
    project_dir: Path,
    file_path: str,
    changes: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Apply FIND/REPLACE edits to a file."""
    full_path = project_dir / file_path
    if not full_path.exists():
        return {"file": file_path, "success": False,
                "error": f"File not found: {file_path}"}

    original = full_path.read_text(encoding="utf-8")
    content = original
    applied = []

    for i, change in enumerate(changes):
        find_text = change.get("find", "")
        replace_text = change.get("replace", "")

        if not find_text:
            continue

        if find_text not in content:
            return {
                "file": file_path,
                "success": False,
                "error": f"Change {i + 1}: Could not find matching text in file",
            }

        content = content.replace(find_text, replace_text, 1)
        applied.append({"find": find_text[:80], "replace": replace_text[:80]})

    # Generate diff
    diff = "\n".join(difflib.unified_diff(
        original.splitlines(),
        content.splitlines(),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    ))

    # Write the file
    full_path.write_text(content, encoding="utf-8")

    return {
        "file": file_path,
        "success": True,
        "changes": len(applied),
        "diff": diff[:1000],
        "size_before": len(original),
        "size_after": len(content),
    }
