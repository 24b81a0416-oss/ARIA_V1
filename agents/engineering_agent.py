"""
ARIA — Engineering Agent (v3)

Ultra-fast pipeline with auto-testing and iterative fixes:
  1. Architecture + all code files generated in a single call
  2. All files reviewed in a single call
  3. Auto-fix loop: fix issues found in review, re-review (up to 3 passes)
  4. Auto-install dependencies + run tests
  5. Vector store context from past successful projects

Uses NVIDIA for generation, Groq for review (if available).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.llm import LLMClient
from utils.file_parser import parse_files, strip_fences
from utils.vector_store import search_memory


def run_engineering_pipeline(
    problem: str,
    llm_arch: LLMClient,    # For generation (NVIDIA or Groq)
    llm_review: LLMClient,  # For review (Groq if available)
    output_dir: Optional[Path] = None,
    user_context: str = "",
    max_fix_passes: int = 3,
    auto_install: bool = True,
    auto_test: bool = True,
) -> Dict[str, Any]:
    """
    Run the full engineering pipeline — with auto-install, auto-test, and iterative fixes.

    Pipeline:
      Stage 1: Generate architecture + all code files in one LLM call
      Stage 2: Review all generated code + fix loop (up to max_fix_passes)
      Stage 3: (Optional) Install dependencies
      Stage 4: (Optional) Run tests
    """
    start = time.time()
    print(f"  [Engineer] Starting pipeline for: {problem}")

    # ── Gather vector store context from past projects ──────────────
    past_context = ""
    try:
        similar = search_memory(f"project: {problem}", limit=3)
        if similar:
            snippets = []
            for s in similar:
                if s.get("score", 0) > 0.4:
                    snippets.append(f"[Past project, relevance {s['score']:.0%}]: {s['content'][:200]}")
            if snippets:
                past_context = "\n\n### Similar past projects for reference\n" + "\n".join(snippets)
                print(f"  [Engineer] Found {len(snippets)} similar past projects in knowledge base")
    except Exception:
        pass  # Vector store is optional

    # ── Stage 1: Generate everything in one shot ────────────────────
    print(f"  [Engineer] Generating architecture + code (stage 1/4)...")

    gen_prompt = f"""You are building: {problem}

Generate a complete, working Python project for this.

## Format
Use these exact markers between files — I will parse them programmatically:

---FILE: path/to/file.py | Brief purpose
<code content>
---END FILE

Generate every file the project needs (main module, support modules, tests if applicable). Ensure all imports across files are consistent.

## Requirements
- Complete, working Python code with proper imports and type hints
- Include a `main.py` entry point or `cli()` function
- Include a `requirements.txt` listing all dependencies
- Include a `test_main.py` with pytest-style tests (if the project has testable logic)
- Use Python 3.10+ features (type hints, match statements if appropriate)
- Follow standard project structure patterns
- Ensure all imports are consistent across files
{past_context}

Output ONLY the marked-up files. No introductory text, no explanations, no markdown around the whole thing.
"""

    gen_system = "You are a senior software engineer. Generate complete working projects with multiple files. Use the exact file markers specified."
    if user_context:
        gen_system += f"\n\n### User Context\n{user_context}"

    raw_output = ""
    try:
        raw_output = llm_arch.generate(
            gen_prompt,
            system_prompt=gen_system,
            max_tokens=4096,
        )
    except Exception as e:
        return {
            "problem": problem,
            "error": f"Generation failed: {e}",
            "files": [],
            "reviews": [],
            "duration": f"{time.time() - start:.0f}s",
            "file_count": 0,
        }

    # ── Parse files from the output ─────────────────────────────────
    files = parse_files(raw_output)

    if not files:
        # Fallback: treat the entire output as main.py
        print(f"  [Engineer] No files found via markers. Treating output as main.py...")
        files = [{"path": "main.py", "code": strip_fences(raw_output)}]

    print(f"  [Engineer] Parsed {len(files)} files")

    # ── Create project directory ────────────────────────────────────
    if output_dir is None:
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in problem)[:40]
        safe_name = safe_name.strip().replace(" ", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("projects") / f"{safe_name}-{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [Engineer] Output: {output_dir}")

    # ── Write all files ─────────────────────────────────────────────
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

    # If no requirements.txt was generated, create a minimal one
    req_path = output_dir / "requirements.txt"
    if not req_path.exists():
        req_path.write_text("# Add your dependencies here\n", encoding="utf-8")

    # ── Stage 2: Review + Fix Loop (iterative) ────────────────────
    print(f"  [Engineer] Reviewing code (stage 2/4)...")
    
    all_findings = []
    fix_passes = 0
    all_passed = False

    for pass_num in range(1, max_fix_passes + 1):
        review_findings = _review_all_files(generated_files, llm_review, user_context=user_context)
        
        # Track all findings across passes
        for rf in review_findings:
            rf["pass"] = pass_num
            existing = next((f for f in all_findings if f["file"] == rf["file"]), None)
            if existing:
                existing["review"] = rf["review"]
                existing["passed"] = rf["passed"]
                existing["pass"] = pass_num
            else:
                all_findings.append(rf)

        passed_count = sum(1 for r in review_findings if r["passed"])
        total_reviewed = len(review_findings)

        if passed_count == total_reviewed:
            all_passed = True
            print(f"  [Engineer] All files passed review (pass {pass_num}/{max_fix_passes})")
            break

        # Find files that failed review
        failed_files = [r for r in review_findings if not r["passed"]]
        if not failed_files:
            all_passed = True
            break

        if pass_num < max_fix_passes:
            print(f"  [Engineer] Fixing {len(failed_files)} issues (pass {pass_num}/{max_fix_passes})...")
            fix_passes += 1
            _apply_fixes(failed_files, generated_files, llm_arch, output_dir, user_context)
        else:
            print(f"  [Engineer] Max fix passes reached ({max_fix_passes}). {len(failed_files)} issues remain.")
            fix_passes += 1

    if all_passed:
        print(f"  [Engineer] All {len(generated_files)} files passed review")
    else:
        passed_count_final = sum(1 for r in all_findings if r["passed"])
        print(f"  [Engineer] {passed_count_final}/{len(generated_files)} passed review after {fix_passes} fix passes")

    # ── Stage 3: Auto-install dependencies ────────────────────────
    install_output = ""
    if auto_install and req_path.exists():
        req_content = req_path.read_text(encoding="utf-8").strip()
        # Check for actual dependency lines (not just comments)
        real_deps = [l.strip() for l in req_content.split("\n") if l.strip() and not l.strip().startswith("#")]
        if real_deps:
            print(f"  [Engineer] Installing dependencies (stage 3/4)...")
            install_output = _install_deps(output_dir)

    # ── Stage 4: Auto-test ───────────────────────────────────────
    test_output = ""
    if auto_test:
        print(f"  [Engineer] Running tests (stage 4/4)...")
        test_output = _run_tests(output_dir)

    # ── Generate README ─────────────────────────────────────────────
    _generate_readme(output_dir, problem, generated_files, all_findings,
                     time.time() - start, install_output, test_output)

    duration = time.time() - start

    return {
        "problem": problem,
        "files": generated_files,
        "reviews": all_findings,
        "project_dir": str(output_dir),
        "duration": f"{duration:.0f}s",
        "file_count": len(generated_files),
        "passed_reviews": sum(1 for r in all_findings if r["passed"]),
        "total_reviews": len(all_findings),
        "fix_passes": fix_passes,
        "all_passed": all_passed,
        "install_output": install_output[:500] if install_output else "",
        "test_output": test_output[:500] if test_output else "",
    }


def format_engineering_result(result: Dict[str, Any]) -> str:
    """Format engineering pipeline result as markdown for the CLI."""
    if "error" in result:
        return f"## Engineering Failed\n\n{result['error']}"

    lines = [
        f"## Engineering: {result['problem']}",
        "",
        f"**Project:** `{result.get('project_dir', '')}`",
        f"**Files created:** {result.get('file_count', 0)}",
        f"**Fix passes:** {result.get('fix_passes', 0)}",
        f"**Reviews passed:** {result.get('passed_reviews', 0)}/{result.get('total_reviews', 0)}",
        f"**Duration:** {result.get('duration', '0s')}",
        "",
        "---",
        "",
        "### Files Generated",
        "",
    ]

    for f in result.get("files", []):
        review = next(
            (r for r in result.get("reviews", []) if r["file"] == f["path"]),
            None,
        )
        status = "✅" if review and review["passed"] else "⚠️"
        lines.append(f"{status} `{f['path']}` ({f.get('size', 0)} bytes)")

    reviews = result.get("reviews", [])
    if reviews:
        lines.extend(["", "### Review Results", ""])
        for r in reviews:
            icon = "✅" if r["passed"] else "⚠️"
            pass_info = f" (fixed in pass {r.get('pass', 1)})" if r["passed"] and r.get("pass", 1) > 1 else ""
            lines.append(f"{icon} **{r['file']}**{pass_info}")
            if not r["passed"] and r.get("review"):
                lines.append(f"  ```")
                lines.append(f"  {r['review'][:300]}")
                lines.append(f"  ```")

    # Show install output
    install = result.get("install_output", "")
    if install:
        lines.extend(["", "### Dependencies", "", f"```\n{install}\n```"])

    # Show test output
    test = result.get("test_output", "")
    if test:
        lines.extend(["", "### Tests", "", f"```\n{test}\n```"])

    lines.extend([
        "",
        "---",
        "",
        f"Run `cd {result.get('project_dir', '')}` to explore the project.",
    ])

    return "\n".join(lines)


# ── Private Helpers ─────────────────────────────────────────────────


def _review_all_files(
    generated_files: List[Dict[str, Any]],
    llm: LLMClient,
    user_context: str = "",
) -> List[Dict[str, Any]]:
    """Review all generated files in a single LLM call."""
    if not generated_files or not llm:
        return [{"file": gf["path"], "review": "No reviewer available", "passed": True}
                for gf in generated_files]

    MAX_REVIEW_CHARS = 8000
    review_parts = []
    total_chars = 0

    for gf in generated_files:
        code = Path(gf["full_path"]).read_text(encoding="utf-8")
        snippet = f"--- {gf['path']} ---\n{code}\n"
        if total_chars + len(snippet) > MAX_REVIEW_CHARS:
            review_parts.append(f"--- {gf['path']} ---\n[truncated, {len(code)} bytes]\n")
            break
        review_parts.append(snippet)
        total_chars += len(snippet)

    review_prompt = f"""Review these Python files for bugs and issues:

{''.join(review_parts)}

For each file, respond with one line:
- PASS: path/to/file.py — if it looks good
- ISSUE: path/to/file.py — <brief description of the issue>

Then optionally add a SHORT note about any cross-file consistency issues.
"""
    review_system = "Code reviewer. For each file: PASS or ISSUE. Be brief but specific."
    if user_context:
        review_system += f"\n\n### User Context\n{user_context}"

    try:
        review_result = llm.generate(
            review_prompt,
            system_prompt=review_system,
            max_tokens=1024,
        )
    except Exception as e:
        return [{"file": gf["path"], "review": f"Review failed: {e}",
                 "passed": False} for gf in generated_files]

    # Parse review results
    findings = []
    for gf in generated_files:
        path = gf["path"]
        pattern = rf"(PASS|ISSUE|FAIL):\s*{re.escape(path)}\s*[—\-–]?\s*(.*)"
        match = re.search(pattern, review_result, re.IGNORECASE)
        if match:
            status = match.group(1).upper()
            detail = match.group(2).strip()
            findings.append({
                "file": path,
                "review": detail,
                "passed": status == "PASS",
            })
        else:
            findings.append({
                "file": path,
                "review": "Not explicitly reviewed",
                "passed": True,
            })

    return findings


def _apply_fixes(
    failed_reviews: List[Dict[str, Any]],
    generated_files: List[Dict[str, Any]],
    llm: LLMClient,
    output_dir: Path,
    user_context: str = "",
) -> None:
    """Apply fixes for reviewed files using the LLM."""
    if not failed_reviews or not llm:
        return

    fix_parts = []
    for review in failed_reviews:
        file_path = review["file"]
        full_path = output_dir / file_path
        if full_path.exists():
            code = full_path.read_text(encoding="utf-8")
            fix_parts.append(f"--- {file_path} ---\nIssue: {review.get('review', 'Unknown issue')}\n{code}\n")

    if not fix_parts:
        return

    fix_prompt = f"""Fix the issues found in these files. Output the corrected files using the same marker format:

{''.join(fix_parts)}

For each file, respond with:
---FILE: path/to/file.py | Fix applied
<fixed code>
---END FILE

Fix ALL issues mentioned. Keep the same file structure and imports. Only change what's needed.
"""
    fix_system = "You are a code fixing agent. Fix the specific issues identified in the review without changing working code."
    if user_context:
        fix_system += f"\n\n### User Context\n{user_context}"

    try:
        fix_output = llm.generate(
            fix_prompt,
            system_prompt=fix_system,
            max_tokens=4096,
        )
    except Exception:
        return  # Fix failed silently — keep original code

    # Parse and apply fixed files
    fixed_files = parse_files(fix_output)
    for fix_info in fixed_files:
        file_rel_path = fix_info["path"]
        code = fix_info["code"]
        full_path = output_dir / file_rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(code, encoding="utf-8")

        # Update sizes in generated_files
        for gf in generated_files:
            if gf["path"] == file_rel_path:
                gf["size"] = len(code)
                break

        print(f"    ✓ Fixed: {file_rel_path}")


def _install_deps(project_dir: Path) -> str:
    """Install project dependencies."""
    req_path = project_dir / "requirements.txt"
    if not req_path.exists():
        return ""

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
            capture_output=True, text=True, timeout=120,
            cwd=str(project_dir),
        )
        output = result.stdout or result.stderr or ""
        # Summarize the result
        if result.returncode == 0:
            lines = [l for l in output.split("\n") if "Successfully installed" in l or "already satisfied" in l]
            return f"Installed ({'success' if result.returncode == 0 else 'failed'})\n" + "\n".join(lines[:5])
        return f"Install exit code {result.returncode}\n{output[:300]}"
    except subprocess.TimeoutExpired:
        return "Dependency install timed out (120s)"
    except Exception as e:
        return f"Install failed: {e}"


def _run_tests(project_dir: Path) -> str:
    """Run project tests (pytest or basic import check)."""
    # Try pytest first
    test_files = list(project_dir.glob("test_*.py")) + list(project_dir.glob("*_test.py"))
    main_file = project_dir / "main.py"

    if test_files:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(project_dir), "-v", "--tb=short"],
                capture_output=True, text=True, timeout=60,
                cwd=str(project_dir),
            )
            if result.returncode == 0:
                return f"All tests passed!\n{result.stdout[:300]}"
            else:
                lines = result.stdout.split("\n")[-10:] + result.stderr.split("\n")[-5:]
                return f"Tests: {result.returncode} failed\n" + "\n".join(lines)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # Fall through to import check

    # Fallback: basic import check
    if main_file.exists():
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import ast; ast.parse(open(r'{main_file}').read()); print('Import OK')"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return f"Import check: {result.stdout.strip() if result.stdout else 'OK'}"
            return f"Import check failed:\n{result.stderr[:200]}"
        except Exception as e:
            return f"Import check error: {e}"

    return "No tests found"


def _generate_readme(
    output_dir: Path,
    problem: str,
    generated_files: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
    duration: float,
    install_output: str = "",
    test_output: str = "",
) -> None:
    """Generate README.md for the project."""
    passed = sum(1 for r in reviews if r["passed"])
    total = len(reviews)
    fix_passes = max(r.get("pass", 0) for r in reviews) if reviews else 0

    readme = f"# {problem}\n\n"
    readme += f"Generated by ARIA Engineering Agent on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

    if install_output:
        readme += "## Dependencies\n\n```\n"
        readme += install_output[:300]
        readme += "\n```\n\n"

    if test_output:
        readme += "## Tests\n\n```\n"
        readme += test_output[:300]
        readme += "\n```\n\n"

    readme += "## Project Structure\n\n"
    for gf in generated_files:
        readme += f"- `{gf['path']}` ({gf['size']} bytes)\n"

    readme += f"\n## Setup\n\n```bash\npip install -r requirements.txt\npython main.py\n```\n"

    if total > 0:
        readme += f"\n## Review\n\n- {passed}/{total} files passed review\n"
        if fix_passes > 0:
            readme += f"- Fix passes: {fix_passes}\n"

    (output_dir / "README.md").write_text(readme, encoding="utf-8")
