"""
ARIA — Codebase Scanner

Scans and analyzes project structure to build a comprehensive context map.
Used by the Editor Agent to understand existing code before making changes.

Capabilities:
  - Walk project tree and classify files
  - Identify project type (Python, Node, etc.)
  - Extract imports, dependencies, and conventions
  - Generate structured context for LLM-driven edits
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Files/directories to skip during scanning
SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules",
    ".venv", "venv", "env", ".env", "dist", "build",
    ".next", ".nuxt", ".turbo", "target", "bin", "obj",
    "projects", ".vscode", ".idea", "coverage", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}
SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".msi", ".deb", ".rpm",
    ".o", ".a", ".lib",
    ".mp3", ".mp4", ".avi", ".mov",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".csv", ".tsv", ".jsonl",
}
MAX_FILE_SIZE = 100 * 1024  # 100KB
MAX_FILES_TO_READ = 15       # Max files to read for content analysis


def scan_project(project_dir: str | Path, max_depth: int = 6) -> Dict[str, Any]:
    """
    Scan a project directory and produce a comprehensive context map.

    Args:
        project_dir: Path to the project root
        max_depth: Maximum directory depth to traverse

    Returns:
        Dict with: project_type, file_tree, imports, dependencies,
                   conventions, key_files, summary
    """
    project_dir = Path(project_dir).resolve()
    if not project_dir.exists():
        return {"error": f"Project directory not found: {project_dir}"}
    if not project_dir.is_dir():
        return {"error": f"Not a directory: {project_dir}"}

    # ── Step 1: Classify project type ────────────────────────────────
    project_type = _detect_project_type(project_dir)
    dependencies = _extract_dependencies(project_dir, project_type)

    # ── Step 2: Walk the file tree ───────────────────────────────────
    file_tree = _build_file_tree(project_dir, max_depth)
    source_files = _find_source_files(project_dir)

    # ── Step 3: Read key files for content analysis ──────────────────
    key_files = _identify_key_files(project_dir, project_type, source_files)
    file_contents = _read_key_files(project_dir, key_files)

    # ── Step 4: Extract imports and conventions ──────────────────────
    imports = _extract_imports(file_contents, project_type)
    conventions = _extract_conventions(file_contents, project_type)

    # ── Step 5: Build summary ────────────────────────────────────────
    summary = _build_summary(project_dir, project_type, file_tree,
                             dependencies, key_files, source_files)

    return {
        "project_dir": str(project_dir),
        "project_name": project_dir.name,
        "project_type": project_type,
        "dependencies": dependencies,
        "file_tree": file_tree,
        "source_files": [str(f.relative_to(project_dir)) for f in source_files],
        "key_files": key_files,
        "file_contents": file_contents,
        "imports": imports,
        "conventions": conventions,
        "summary": summary,
    }


def format_project_context(scanner_result: Dict[str, Any]) -> str:
    """Format scanner result into a concise markdown context block for the LLM."""
    if "error" in scanner_result:
        return f"**Error:** {scanner_result['error']}"

    lines = [
        f"## Project: {scanner_result['project_name']}",
        f"**Type:** {scanner_result['project_type']}",
        f"**Directory:** `{scanner_result['project_dir']}`",
        "",
    ]

    # Dependencies
    deps = scanner_result.get("dependencies", {})
    if deps:
        lines.append("### Dependencies")
        lines.append("")
        for dep_type, dep_list in deps.items():
            if dep_list:
                lines.append(f"**{dep_type}:** `{'`, `'.join(dep_list[:20])}`")
        lines.append("")

    # Key files
    key_files = scanner_result.get("key_files", [])
    if key_files:
        lines.append("### Key Files")
        lines.append("")
        for kf in key_files[:10]:
            lines.append(f"- `{kf['path']}` ({kf.get('purpose', 'unknown')}) — {kf.get('size', 0)} bytes")
        lines.append("")

    # Imports overview
    imports = scanner_result.get("imports", {})
    if imports:
        lines.append("### Import Patterns")
        lines.append("")
        for lib, count in sorted(imports.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"- `{lib}` ({count}x)")
        lines.append("")

    # Conventions
    conventions = scanner_result.get("conventions", {})
    if conventions:
        lines.append("### Conventions")
        lines.append("")
        for key, val in conventions.items():
            if val:
                lines.append(f"- **{key}:** {val}")
        lines.append("")

    # Summary
    summary = scanner_result.get("summary", "")
    if summary:
        lines.append(f"### Summary\n\n{summary}\n")

    return "\n".join(lines)


# ── Private: Project Detection ───────────────────────────────────────

def _detect_project_type(project_dir: Path) -> str:
    """Detect the project type based on config files."""
    indicators = {
        ("package.json",): "Node.js / JavaScript / TypeScript",
        ("pyproject.toml",): "Python (pyproject.toml)",
        ("requirements.txt", "setup.py", "setup.cfg"): "Python",
        ("Cargo.toml",): "Rust",
        ("go.mod",): "Go",
        ("pom.xml", "build.gradle", "build.gradle.kts"): "Java",
        ("Gemfile",): "Ruby",
        ("mix.exs",): "Elixir",
        ("Project.toml", "Manifest.toml"): "Julia",
        ("composer.json",): "PHP",
        ("Makefile", "CMakeLists.txt"): "C/C++",
        ("index.html",): "Static Web",
    }

    for files, ptype in indicators.items():
        if any((project_dir / f).exists() for f in files):
            return ptype

    # Check for common source dirs
    if (project_dir / "src").is_dir() or (project_dir / "lib").is_dir():
        py_files = list(project_dir.rglob("*.py"))
        js_files = list(project_dir.rglob("*.js"))
        ts_files = list(project_dir.rglob("*.ts"))
        if py_files and not js_files and not ts_files:
            return "Python"
        if (js_files or ts_files) and not py_files:
            return "JavaScript / TypeScript"

    return "Unknown"


def _extract_dependencies(project_dir: Path, project_type: str) -> Dict[str, List[str]]:
    """Extract dependencies from config files."""
    deps: Dict[str, List[str]] = {"runtime": [], "dev": []}

    try:
        if "Python" in project_type:
            req_file = project_dir / "requirements.txt"
            if req_file.exists():
                for line in req_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        deps["runtime"].append(line.split("==")[0].split(">=")[0].split("<")[0].strip())

            pyproject = project_dir / "pyproject.toml"
            if pyproject.exists():
                for line in pyproject.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line.startswith('"') and "=" in line:
                        pkg = line.split("=")[0].strip().strip('"')
                        if pkg and pkg not in ("python",):
                            deps["runtime"].append(pkg)

    except Exception:
        pass

    # Deduplicate
    deps["runtime"] = sorted(set(deps["runtime"]))
    deps["dev"] = sorted(set(deps["dev"]))
    return deps


# ── Private: File Tree ──────────────────────────────────────────────

def _build_file_tree(project_dir: Path, max_depth: int) -> List[Dict[str, Any]]:
    """Build a structured file tree."""
    tree = []

    def walk(current: Path, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda x: (x.is_file(), x.name))
        except PermissionError:
            return

        for entry in entries:
            if entry.name.startswith(".") and entry.name != ".env":
                continue
            if entry.is_dir():
                if entry.name in SKIP_DIRS:
                    continue
                walk(entry, depth + 1)
                tree.append({
                    "name": str(entry.relative_to(project_dir)),
                    "type": "dir",
                    "depth": depth,
                })
            elif entry.is_file():
                if entry.suffix in SKIP_EXTENSIONS:
                    continue
                if entry.stat().st_size > MAX_FILE_SIZE:
                    continue
                tree.append({
                    "name": str(entry.relative_to(project_dir)),
                    "type": "file",
                    "size": entry.stat().st_size,
                    "depth": depth,
                })

    walk(project_dir, 0)
    return tree


def _find_source_files(project_dir: Path) -> List[Path]:
    """Find all readable source files."""
    files = []
    for ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java",
                ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".swift", ".kt",
                ".sh", ".bash", ".zsh", ".yaml", ".yml", ".json", ".toml",
                ".cfg", ".ini", ".md", ".rst", ".html", ".css", ".scss"]:
        for f in project_dir.rglob(f"*{ext}"):
            # Skip hidden dirs, venv, node_modules, etc.
            if any(part.startswith(".") or part in SKIP_DIRS for part in f.relative_to(project_dir).parts):
                continue
            if f.stat().st_size > MAX_FILE_SIZE:
                continue
            files.append(f)
    return sorted(files)


# ── Private: Key Files ──────────────────────────────────────────────

def _identify_key_files(
    project_dir: Path,
    project_type: str,
    source_files: List[Path],
) -> List[Dict[str, Any]]:
    """Identify the most important files for understanding the project."""
    key_files = []

    # Config files
    config_names = [
        "package.json", "requirements.txt", "setup.py", "setup.cfg",
        "pyproject.toml", "Cargo.toml", "go.mod", "Makefile",
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        ".env.example", ".env.sample", ".gitignore",
        "tsconfig.json", "vite.config.ts", "vite.config.js",
        "webpack.config.js", "next.config.js", "nuxt.config.ts",
        "README.md", "CONTRIBUTING.md", "LICENSE",
    ]
    for name in config_names:
        path = project_dir / name
        if path.exists() and path.stat().st_size < MAX_FILE_SIZE:
            key_files.append({
                "path": name,
                "purpose": _file_purpose(name),
                "size": path.stat().st_size,
            })

    # Entry points
    entry_names = [
        "main.py", "app.py", "cli.py", "index.py", "server.py",
        "index.js", "index.ts", "index.jsx", "index.tsx",
        "main.js", "main.ts", "app.js", "app.ts",
        "src/main.py", "src/index.ts", "src/app.ts",
        "lib/main.dart", "bin/main.dart",
    ]
    for name in entry_names:
        path = project_dir / name
        if path.exists() and path.stat().st_size < MAX_FILE_SIZE:
            key_files.append({
                "path": name,
                "purpose": "Entry point",
                "size": path.stat().st_size,
            })

    # Source files (limit to most important)
    seen = {kf["path"] for kf in key_files}
    for sf in source_files:
        rel = str(sf.relative_to(project_dir))
        if rel not in seen and len(key_files) < MAX_FILES_TO_READ + 5:
            key_files.append({
                "path": rel,
                "purpose": _classify_file(rel),
                "size": sf.stat().st_size,
            })
            seen.add(rel)

    return key_files[:MAX_FILES_TO_READ + 5]


def _file_purpose(name: str) -> str:
    """Guess the purpose of a config file."""
    purposes = {
        "package.json": "NPM dependencies & scripts",
        "requirements.txt": "Python dependencies",
        "setup.py": "Python package config",
        "setup.cfg": "Python package config (legacy)",
        "pyproject.toml": "Python project config",
        "Cargo.toml": "Rust project config",
        "go.mod": "Go module config",
        "Makefile": "Build automation",
        "Dockerfile": "Container build",
        "docker-compose.yml": "Container orchestration",
        ".env.example": "Environment variable template",
        ".gitignore": "Git ignore rules",
        "tsconfig.json": "TypeScript config",
        "vite.config.ts": "Vite build config",
        "README.md": "Project documentation",
        "LICENSE": "Software license",
    }
    return purposes.get(name, "Source file")


def _classify_file(rel_path: str) -> str:
    """Classify a source file by its path context."""
    if "/test" in rel_path or "/tests/" in rel_path or rel_path.startswith("test_"):
        return "Tests"
    if "/migrations/" in rel_path:
        return "Database migration"
    if "/fixtures/" in rel_path or "/data/" in rel_path:
        return "Data / fixtures"
    if "/config/" in rel_path:
        return "Configuration"
    if "/utils/" in rel_path or "/helpers/" in rel_path:
        return "Utilities"
    if "/middleware/" in rel_path:
        return "Middleware"
    if "/routes/" in rel_path or "/views/" in rel_path:
        return "Routes / views"
    if "/models/" in rel_path:
        return "Models"
    if "/controllers/" in rel_path:
        return "Controllers"
    if "/services/" in rel_path:
        return "Services"
    if "/components/" in rel_path:
        return "Components"
    if "/api/" in rel_path:
        return "API"
    return "Source module"


def _read_key_files(project_dir: Path, key_files: List[Dict[str, Any]]) -> Dict[str, str]:
    """Read content of key files (truncated for large files)."""
    contents = {}
    for kf in key_files:
        try:
            path = project_dir / kf["path"]
            if path.exists() and path.stat().st_size > 0:
                text = path.read_text(encoding="utf-8", errors="ignore")
                contents[kf["path"]] = text[:5000]  # Truncate to 5KB
        except Exception:
            pass
    return contents


# ── Private: Imports & Conventions ──────────────────────────────────

def _extract_imports(
    file_contents: Dict[str, str],
    project_type: str,
) -> Dict[str, int]:
    """Extract import patterns from file contents."""
    import_counts: Dict[str, int] = {}

    for path, content in file_contents.items():
        if path.endswith(".py"):
            for match in re.finditer(r"^(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", content, re.MULTILINE):
                module = match.group(1).split(".")[0]
                import_counts[module] = import_counts.get(module, 0) + 1
        elif path.endswith((".js", ".ts", ".jsx", ".tsx")):
            for match in re.finditer(r"(?:import|require)\s*\(?['\"]([^'\"/]+)", content):
                lib = match.group(1).split("/")[0]
                if lib and not lib.startswith("."):
                    import_counts[lib] = import_counts.get(lib, 0) + 1

    return dict(sorted(import_counts.items(), key=lambda x: -x[1]))


def _extract_conventions(
    file_contents: Dict[str, str],
    project_type: str,
) -> Dict[str, str]:
    """Extract code conventions from file contents."""
    conventions: Dict[str, str] = {}

    if "Python" in project_type:
        # Check for type hints
        type_hints = sum(1 for c in file_contents.values() if ":" in c and "def " in c)
        conventions["type_hints"] = "Yes" if type_hints > 2 else "No"

        # Check for async usage
        async_count = sum(1 for c in file_contents.values() if "async def" in c)
        conventions["async"] = f"Used in {async_count} functions" if async_count else "Not used"

        # Check for classes vs functions
        class_count = sum(1 for c in file_contents.values() for _ in re.finditer(r"^class\s", c, re.MULTILINE))
        def_count = sum(1 for c in file_contents.values() for _ in re.finditer(r"^def\s", c, re.MULTILINE))
        conventions["style"] = f"Classes: {class_count}, Functions: {def_count}"

        # Check for docstrings
        docstring_count = sum(1 for c in file_contents.values() for _ in re.finditer(r'""".*?"""', c, re.DOTALL))
        conventions["docstrings"] = f"Found in {docstring_count} places"

        # Check for error handling
        try_count = sum(1 for c in file_contents.values() for _ in re.finditer(r"\btry\b", c))
        conventions["error_handling"] = f"try/except used in {try_count} places"

    elif "TypeScript" in project_type or "JavaScript" in project_type:
        ts_count = sum(1 for _ in file_contents if _.endswith(".ts") or _.endswith(".tsx"))
        conventions["typescript"] = f"{'Yes' if ts_count > 0 else 'No'} ({ts_count} files)"

        react_count = sum(1 for c in file_contents.values() if "React" in c or "react" in c)
        conventions["framework"] = f"React detected" if react_count else "Unknown"

    return conventions


# ── Private: Summary ────────────────────────────────────────────────

def _build_summary(
    project_dir: Path,
    project_type: str,
    file_tree: List[Dict[str, Any]],
    dependencies: Dict[str, List[str]],
    key_files: List[Dict[str, Any]],
    source_files: List[Path],
) -> str:
    """Build a concise project summary."""
    source_count = len(source_files)
    total_size = sum(f.stat().st_size for f in source_files[:100])

    dirs = [e for e in file_tree if e["type"] == "dir"]
    files = [e for e in file_tree if e["type"] == "file"]
    py_files = len([s for s in source_files if s.suffix == ".py"])
    js_files = len([s for s in source_files if s.suffix in (".js", ".jsx")])
    ts_files = len([s for s in source_files if s.suffix in (".ts", ".tsx")])

    top_deps = dependencies.get("runtime", [])[:8]
    dep_str = ", ".join(top_deps) if top_deps else "None detected"

    return (
        f"{project_type} project with {source_count} source files "
        f"({py_files} Python, {js_files} JS, {ts_files} TypeScript) "
        f"across {len(dirs)} directories. "
        f"Total source size: ~{total_size // 1024}KB. "
        f"Key dependencies: {dep_str}. "
        f"Scanned {len(key_files)} key files for context."
    )
