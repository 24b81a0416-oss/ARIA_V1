"""
ARIA — Skill Manager

Modular skill system inspired by anthropics/skills.
Skills are SKILL.md files with YAML frontmatter + Markdown instructions.

Each skill has:
  - name: kebab-case identifier
  - description: What it does
  - version: Semver
  - instructions: The body — system prompt extensions for the LLM

Usage:
  skill list                    — List all available skills
  skill load [name]             — Load a skill's instructions
  skill show [name]             — Show full skill details
  skill create [name]           — Create a new skill interactively
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SKILLS_DIR = Path(__file__).parent.parent / "skills"

SKILL_TEMPLATE = """---
name: {name}
description: {description}
version: 1.0.0
---

# {name}

## Purpose
{description}

## Instructions

<!-- Add your instructions here. These will be injected into the system prompt
     when this skill is activated. Be specific about:
     - When this skill should be used
     - What the AI should do step by step
     - Output format expectations
     - Any constraints or guidelines
-->

## Examples

<!-- Optional: Add example prompts and expected outputs -->

"""


def list_skills() -> List[Dict[str, Any]]:
    """List all available skills with their metadata."""
    skills_dir = SKILLS_DIR
    if not skills_dir.exists():
        return []

    skills = []
    for skill_dir in sorted(skills_dir.iterdir()):
        if skill_dir.is_dir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                meta = _parse_skill_meta(skill_file.read_text(encoding="utf-8"))
                if meta:
                    skills.append(meta)

    return skills


def load_skill(name: str) -> Optional[Dict[str, Any]]:
    """Load a skill by name. Returns full skill dict including instructions."""
    skill_path = _find_skill_path(name)
    if not skill_path:
        return None

    content = skill_path.read_text(encoding="utf-8")
    meta = _parse_skill_meta(content)
    if not meta:
        return None

    instructions = _extract_instructions(content)
    meta["instructions"] = instructions
    return meta


def create_skill(name: str, description: str = "") -> Optional[Path]:
    """Create a new skill with the given name and description."""
    skills_dir = SKILLS_DIR
    skills_dir.mkdir(parents=True, exist_ok=True)

    skill_dir = skills_dir / name
    if skill_dir.exists():
        return None  # Skill already exists

    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"

    content = SKILL_TEMPLATE.format(
        name=name,
        description=description or f"Skill for {name}",
    )
    skill_file.write_text(content, encoding="utf-8")
    return skill_file


def get_instructions_for_task(task: str, llm_client=None) -> Tuple[str, List[str]]:
    """
    Automatically select and load skills relevant to a given task.

    Returns:
        (combined_instructions, skill_names_used)
    """
    skills = list_skills()
    if not skills:
        return "", []

    task_lower = task.lower()
    relevant = []

    for skill in skills:
        name = skill.get("name", "")
        desc = skill.get("description", "").lower()
        # Simple keyword matching
        if any(word in task_lower for word in name.lower().split("-")):
            relevant.append(name)
        elif any(word in task_lower for word in desc.split()):
            relevant.append(name)

    # Deduplicate and limit
    relevant = list(dict.fromkeys(relevant))[:3]

    combined = ""
    names_used = []
    for name in relevant:
        skill = load_skill(name)
        if skill and skill.get("instructions"):
            combined += f"\n## Skill: {name}\n{skill['instructions']}\n"
            names_used.append(name)

    return combined.strip(), names_used


def get_skill_context(names: List[str]) -> str:
    """Load skill instructions by name and combine into a context block."""
    parts = []
    for name in names:
        skill = load_skill(name)
        if skill and skill.get("instructions"):
            parts.append(f"## Skill: {name}\n{skill['instructions']}")
    return "\n\n".join(parts)


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_skill_meta(content: str) -> Optional[Dict[str, Any]]:
    """Parse YAML frontmatter from skill content."""
    # Match --- frontmatter -- (simple YAML parser)
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None

    frontmatter = match.group(1)
    meta = {"name": "", "description": "", "version": "1.0.0"}

    for line in frontmatter.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip().strip('"').strip("'")
            if key in meta:
                meta[key] = value

    if not meta["name"]:
        return None

    return meta


def _extract_instructions(content: str) -> str:
    """Extract instructions (content after frontmatter)."""
    match = re.match(r"^---\s*\n.*?\n---\s*\n(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


def _find_skill_path(name: str) -> Optional[Path]:
    """Find the SKILL.md path for a skill by name."""
    skills_dir = SKILLS_DIR
    if not skills_dir.exists():
        return None

    for skill_dir in skills_dir.iterdir():
        if skill_dir.is_dir() and skill_dir.name == name:
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                return skill_file

    return None
