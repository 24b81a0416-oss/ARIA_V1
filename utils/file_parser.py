"""
ARIA — File Parser Utility

Shared parsing logic for extracting files from LLM output.
Used by coder_agent, debugger_agent, and engineering_agent.

Provides:
  - _parse_files: Parse ---FILE: / ---END FILE markers or code blocks
  - _strip_fences: Remove surrounding markdown fences
  - _sanitize_name: Create safe directory names
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# ── File separator constants ─────────────────────────────────────────
FILE_SEP_START = "---FILE:"
FILE_SEP_END = "---END FILE"


def parse_files(raw: str) -> List[Dict[str, str]]:
    """
    Parse multiple files from LLM output.

    Tries three strategies in order:
    1. Custom markers (---FILE: / ---END FILE)
    2. Markdown code blocks with filenames (```python path/to/file.py)
    3. Any markdown code blocks (```python ... ```)

    Returns:
        List of dicts with 'path' and 'code' keys
    """
    # Strategy 1: Custom markers
    files = _parse_with_markers(raw)
    if files:
        return files

    # Strategy 2: Markdown code blocks with filenames
    files = _parse_with_code_blocks(raw)
    if files:
        return files

    return []


def strip_fences(code: str) -> str:
    """Remove surrounding markdown fences if present."""
    code = code.strip()
    if code.startswith("```"):
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline + 1:]
        if code.endswith("```"):
            code = code[:-3].strip()
    return code.strip()


def sanitize_name(description: str) -> str:
    """Create a safe directory name from a description."""
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in description)
    return safe.strip().replace(" ", "-")


# ── Private helpers ──────────────────────────────────────────────────


def _parse_with_markers(raw: str) -> List[Dict[str, str]]:
    """Parse files using ---FILE: / ---END FILE markers.

    Each part looks like:
      path/to/file.py | Description
      <code content>
      ---END FILE
    """
    files = []
    parts = raw.split(FILE_SEP_START)
    for part in parts[1:]:
        end_idx = part.find(FILE_SEP_END)
        if end_idx == -1:
            continue

        section = part[:end_idx].strip()
        first_newline = section.find("\n")
        if first_newline != -1:
            header_line = section[:first_newline].strip()
            code = section[first_newline + 1:].strip()
        else:
            header_line = section
            code = ""

        # Parse header to get file path and optional description
        description = ""
        if "|" in header_line:
            parts_h = header_line.split("|", 1)
            file_path = parts_h[0].strip()
            description = parts_h[1].strip()
        else:
            file_path = header_line.strip()

        code = strip_fences(code)
        if code and file_path:
            files.append({
                "path": file_path,
                "code": code,
                "description": description,
            })

    return files


def _parse_with_code_blocks(raw: str) -> List[Dict[str, str]]:
    """Parse files from markdown code blocks with optional filenames.

    Handles:
      ```python
      code here
      ```

      ```python:path/to/file.py
      code here
      ```

      ```python title="main.py"
      code here
      ```
    """
    files = []
    pattern = r'```(\w*)\s*(?::|title="|path=)?([^"\n]*)?\n(.*?)```'
    matches = re.findall(pattern, raw, re.DOTALL)

    for lang, filename, code in matches:
        code = code.strip()
        if not code:
            continue

        # Determine filename
        if filename and filename.strip():
            fname = filename.strip()
        elif lang in ("python", "py"):
            fname = f"module_{len(files) + 1}.py"
        elif lang:
            fname = f"file_{len(files) + 1}.{lang}"
        else:
            fname = f"file_{len(files) + 1}"

        # Skip shell command blocks without filenames
        SKIP_LANGS = {"bash", "sh", "console", "text", ""}
        if lang in SKIP_LANGS and not filename:
            continue

        files.append({"path": fname, "code": code})

    return files
