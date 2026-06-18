"""
ARIA — Coder Agent

Code generation from architecture specs.
Takes architecture documentation and generates complete, working code files.
Can generate single files, multi-file projects, or specific components.

Commands:
  `code [description]`  — Generate code from description or architecture
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.llm import LLMClient
from utils.file_parser import parse_files, strip_fences, sanitize_name


def run_coding(
    description: str,
    llm: LLMClient,
    architecture: Optional[str] = None,
    output_dir: Optional[Path] = None,
    user_context: str = "",
    existing_context: str = "",
) -> Dict[str, Any]:
    """
    Generate code files from a description or architecture spec.

    Args:
        description: What to build
        llm: LLM client for code generation
        architecture: Optional architecture doc to guide generation
        output_dir: Where to write the generated files
        user_context: User facts from memory

    Returns:
        Dict with: description, files[], file_count, duration
    """
    start = time.time()
    print(f"  [Coder] Generating code for: {description}")

    # Build the prompt with optional architecture context
    arch_context = ""
    if architecture:
        arch_context = f"\n\n## Architecture Reference\n{architecture[:4000]}\n"

    gen_prompt = f"""Generate complete Python code for:

Project: {description}
{arch_context}
{existing_context}

## Format
Use these exact markers between files — I will parse them programmatically:

---FILE: path/to/file.py | Brief purpose
<code content>
---END FILE

Generate every file the project needs. Ensure imports across files are consistent.

## Requirements
- Complete, working Python code with proper imports and type hints
- Include a `main.py` entry point or `cli()` function
- Include a `requirements.txt` listing all dependencies
- Use Python 3.10+ features
- Follow standard project structure patterns
- Add docstrings for all public functions and classes

Output ONLY the marked-up files. No introductory text, no explanations.
"""

    gen_system = (
        "You are a senior software engineer. Generate complete, production-quality "
        "Python code. Use the exact file markers specified."
    )
    if user_context:
        gen_system += f"\n\n### User Context\n{user_context}"

    try:
        raw_output = llm.generate(
            gen_prompt,
            system_prompt=gen_system,
            max_tokens=4096,
        )
    except Exception as e:
        return {
            "description": description,
            "error": f"Code generation failed: {e}",
            "files": [],
            "file_count": 0,
            "duration": f"{time.time() - start:.0f}s",
        }

    # ── Parse files from the output ─────────────────────────────────
    files = parse_files(raw_output)

    if not files:
        print(f"  [Coder] No files found via markers. Treating output as main.py...")
        files = [{"path": "main.py", "code": strip_fences(raw_output)}]

    print(f"  [Coder] Parsed {len(files)} files")

    # ── Write files ──────────────────────────────────────────────────
    if output_dir is None:
        safe_name = sanitize_name(description)[:40]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("projects") / f"{safe_name}-{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    for file_info in files:
        file_rel_path = file_info["path"]
        code = file_info["code"]

        file_full_path = output_dir / file_rel_path
        file_full_path.parent.mkdir(parents=True, exist_ok=True)
        file_full_path.write_text(code, encoding="utf-8")

        generated_files.append({
            "path": str(file_rel_path),
            "size": len(code),
            "full_path": str(file_full_path),
        })
        print(f"    → {file_rel_path} ({len(code)} bytes)")

    # Ensure requirements.txt exists
    req_path = output_dir / "requirements.txt"
    if not req_path.exists():
        req_path.write_text("# Add your dependencies here\n", encoding="utf-8")

    duration = time.time() - start
    print(f"  [Coder] Complete ({duration:.0f}s) — {len(generated_files)} files")

    return {
        "description": description,
        "files": generated_files,
        "project_dir": str(output_dir),
        "duration": f"{duration:.0f}s",
        "file_count": len(generated_files),
    }


def format_coding_result(result: Dict[str, Any]) -> str:
    """Format coding result as markdown for the CLI."""
    if "error" in result:
        return f"## Code Generation Failed\n\n{result['error']}"

    lines = [
        f"## Code: {result['description']}",
        "",
        f"**Files created:** {result.get('file_count', 0)}",
        f"**Project:** `{result.get('project_dir', '')}`",
        f"**Duration:** {result.get('duration', '0s')}",
        "",
        "---",
        "",
        "### Generated Files",
        "",
    ]

    for f in result.get("files", []):
        lines.append(f"- `{f['path']}` ({f.get('size', 0)} bytes)")

    lines.extend([
        "",
        f"Run `cd {result.get('project_dir', '')}` to explore.",
    ])

    return "\n".join(lines)

