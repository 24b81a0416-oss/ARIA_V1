"""
ARIA — Bash Agent

Executes terminal commands with safety features:
  - Blocklist of dangerous commands
  - Timeout (default 60s)
  - Output cap (10KB)
  - Confirmation prompt (bash) vs no-confirm (bash!)

Commands:
  `bash [command]`    — Execute with safety confirmation
  `bash! [command]`   — Execute without confirmation (expert mode)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Safety Blocklist ─────────────────────────────────────────────────

DANGEROUS_PATTERNS: List[str] = [
    # File destruction
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf .",
    "rm /",
    "del /f /s",
    "del /f /q",
    "format ",
    "mkfs",
    "dd if=",
    # System commands
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "init 0",
    "init 6",
    # Privilege escalation
    "sudo ",
    "su ",
    "chmod 777 ",
    "chown ",
    "passwd ",
    # Network dangerous
    "iptables -F",
    "ufw disable",
    "route ",
    "ifconfig ",
    "ip link set ",
    # Windows dangerous
    "reg delete",
    "reg add",
    "regedit",
    "diskpart",
    "bcdedit",
    "bootrec",
]

# Commands that are always allowed (whitelist overrides blocklist for these)
ALWAYS_ALLOWED_PREFIXES: List[str] = [
    "python",
    "python3",
    "pip",
    "pip3",
    "node",
    "npm",
    "npx",
    "yarn",
    "pnpm",
    "cargo",
    "go",
    "rustc",
    "gcc",
    "g++",
    "clang",
    "make",
    "cmake",
    "git",
    "echo",
    "cat",
    "type ",
    "dir",
    "ls",
    "pwd",
    "cd ",
    "mkdir",
    "touch",
    "cp ",
    "copy ",
    "move ",
    "rename ",
    "head ",
    "tail ",
    "less ",
    "more ",
    "find ",
    "grep",
    "findstr",
    "wc ",
    "sort ",
    "uniq ",
    "tee ",
    "curl ",
    "wget ",
    "print",
    "which ",
    "where ",
    "whoami",
    "date",
    "time",
    "hostname",
    "uname",
]

DEFAULT_TIMEOUT = 60  # seconds
MAX_OUTPUT_CHARS = 10_000


# ── Safety Check ─────────────────────────────────────────────────────

def is_command_safe(command: str) -> Tuple[bool, str]:
    """
    Check if a command is safe to execute.

    Dangerous patterns are checked FIRST (regardless of prefix).
    Then allowed prefixes are checked for convenience.

    Returns: (is_safe, reason_if_unsafe)
    """
    cmd_lower = command.strip().lower()

    # Check dangerous patterns FIRST — always, even for pip/python/git commands
    for pattern in DANGEROUS_PATTERNS:
        if pattern in cmd_lower:
            return False, f"Blocked by safety: command contains '{pattern}'"

    # Check if starts with an allowed prefix
    for prefix in ALWAYS_ALLOWED_PREFIXES:
        if cmd_lower.startswith(prefix):
            return True, ""

    return True, ""


# ── Command Execution ────────────────────────────────────────────────

def run_command(
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Execute a command and return the result.

    Args:
        command: The shell command to execute
        timeout: Max seconds to wait (default 60, -1 for no timeout)
        cwd: Working directory (default: current)
        env: Optional environment variables to set

    Returns:
        Dict with: command, return_code, stdout, stderr, duration, error, truncated
    """
    start = time.time()

    # Safety check
    safe, reason = is_command_safe(command)
    if not safe:
        return {
            "command": command,
            "return_code": -1,
            "stdout": "",
            "stderr": reason,
            "duration": f"{time.time() - start:.1f}s",
            "error": reason,
            "truncated": False,
        }

    # Determine shell
    if sys.platform == "win32":
        shell_cmd = ["cmd", "/c", command]
    else:
        shell_cmd = ["sh", "-c", command]

    # Resolve working directory
    work_dir = str(cwd.resolve()) if cwd else os.getcwd()

    # Merge environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    # Add .local/bin to PATH for pip-installed tools
    local_bin = Path.home() / ".local" / "bin"
    if local_bin.exists():
        path = run_env.get("PATH", "")
        run_env["PATH"] = f"{local_bin}{os.pathsep}{path}"

    try:
        proc = subprocess.Popen(
            shell_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=work_dir,
            env=run_env,
            text=True,
            # Don't create a new window on Windows
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        # Read output with timeout
        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout if timeout > 0 else None)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_data, stderr_data = proc.communicate()
            duration = time.time() - start
            return {
                "command": command,
                "return_code": -1,
                "stdout": stdout_data[:MAX_OUTPUT_CHARS] if stdout_data else "",
                "stderr": f"Command timed out after {timeout}s",
                "duration": f"{duration:.1f}s",
                "error": f"Command timed out after {timeout}s",
                "truncated": len(stdout_data or "") > MAX_OUTPUT_CHARS,
            }

        duration = time.time() - start
        return_code = proc.returncode

        # Truncate output if needed
        truncated = False
        if stdout_data and len(stdout_data) > MAX_OUTPUT_CHARS:
            stdout_data = stdout_data[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
            truncated = True
        if stderr_data and len(stderr_data) > MAX_OUTPUT_CHARS:
            stderr_data = stderr_data[:MAX_OUTPUT_CHARS] + "\n... [stderr truncated]"
            truncated = True

        return {
            "command": command,
            "return_code": return_code,
            "stdout": stdout_data or "",
            "stderr": stderr_data or "",
            "duration": f"{duration:.1f}s",
            "error": None if return_code == 0 else f"Exit code {return_code}",
            "truncated": truncated,
        }

    except FileNotFoundError as e:
        return {
            "command": command,
            "return_code": -1,
            "stdout": "",
            "stderr": f"Command not found: {e}",
            "duration": f"{time.time() - start:.1f}s",
            "error": f"Command not found: {e}",
            "truncated": False,
        }
    except Exception as e:
        return {
            "command": command,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "duration": f"{time.time() - start:.1f}s",
            "error": str(e),
            "truncated": False,
        }


# ── Format Result ────────────────────────────────────────────────────

def format_result(result: Dict[str, Any], analyze: bool = False) -> str:
    """
    Format a command result for CLI display.

    Args:
        result: The result dict from run_command
        analyze: If True, include an AI-style analysis of the output

    Returns:
        Formatted string
    """
    lines = []
    cmd = result["command"]

    # Header
    lines.append(f"```bash")
    lines.append(f"$ {cmd}")
    lines.append("```")

    # Output
    stdout = result.get("stdout", "").strip()
    stderr = result.get("stderr", "").strip()
    error = result.get("error")
    duration = result.get("duration", "0s")
    return_code = result.get("return_code", 0)
    truncated = result.get("truncated", False)

    if stdout:
        lines.append("")
        lines.append("**Output:**")
        lines.append("```")
        lines.append(stdout[:3000])  # Cap display as well
        lines.append("```")

    if stderr:
        lines.append("")
        lines.append("**Errors:**")
        lines.append("```")
        lines.append(stderr[:2000])
        lines.append("```")

    if error:
        lines.append("")
        lines.append(f"**⚠️ {error}**")

    # Footer
    status = "✅" if return_code == 0 else "❌"
    lines.append("")
    lines.append(f"_{status} Exit: {return_code} | Duration: {duration}_")
    if truncated:
        lines.append("_Output was truncated (limit: 10KB)_")

    return "\n".join(lines)
