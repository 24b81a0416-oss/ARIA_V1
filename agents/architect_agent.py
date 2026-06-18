"""
ARIA — Architect Agent

System design and architecture planning.
Takes a project description and generates:
  - System architecture overview
  - Component/module breakdown
  - Data flow design
  - Technology stack recommendations
  - Key design decisions with rationale

Can be used standalone or as the first stage in the engineering pipeline.

Commands:
  `architect [description]`  — Generate architecture docs
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.llm import LLMClient


def run_architecture(
    description: str,
    llm: LLMClient,
    output_dir: Optional[Path] = None,
    user_context: str = "",
    existing_context: str = "",
) -> Dict[str, Any]:
    """
    Generate architecture documentation for a project.

    Args:
        description: What to build
        llm: LLM client for generation
        output_dir: Where to save architecture docs (optional)
        user_context: User facts from memory
        existing_context: Previous sub-task results (for multi-agent flows)

    Returns:
        Dict with: description, architecture, components[], tech_stack[],
                   design_decisions[], duration
    """
    start = time.time()
    print(f"  [Architect] Designing architecture for: {description}")

    arch_prompt = f"""Design the architecture for:

Project: {description}

{existing_context}

Produce a comprehensive architecture document with:

## 1. System Overview
- High-level description of the system
- Core purpose and key capabilities

## 2. Architecture Style
- Pattern (microservices, monolith, event-driven, etc.)
- Why this pattern fits

## 3. Component Breakdown
For each major component:
- Name and responsibility
- Key interfaces/APIs
- Dependencies on other components

## 4. Data Flow
- How data moves through the system
- Storage decisions and rationale

## 5. Technology Stack
- Language, framework, database, messaging, etc.
- Alternatives considered and why chosen

## 6. Key Design Decisions
- 3-5 important decisions with rationale
- Trade-offs made

Output in clear markdown with the exact section headers shown above.
"""

    arch_system = "You are a senior software architect. Design clear, practical, well-reasoned architectures. Be specific about components, data flow, and technology choices."
    if user_context:
        arch_system += f"\n\n### User Context\n{user_context}"

    try:
        architecture = llm.generate(
            arch_prompt,
            system_prompt=arch_system,
            max_tokens=3072,
        )
    except Exception as e:
        return {
            "description": description,
            "error": f"Architecture generation failed: {e}",
            "duration": f"{time.time() - start:.0f}s",
        }

    # Parse key sections from the output (for display)
    print(f"  [Architect] Architecture complete ({len(architecture)} chars)")

    # Save to file if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        arch_path = output_dir / "ARCHITECTURE.md"
        header = f"# Architecture: {description}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n\n"
        arch_path.write_text(header + architecture, encoding="utf-8")
        print(f"  [Architect] Saved to {arch_path.name}")

    duration = time.time() - start
    print(f"  [Architect] Complete ({duration:.0f}s)")

    return {
        "description": description,
        "architecture": architecture.strip(),
        "duration": f"{duration:.0f}s",
    }


def format_architecture_result(result: Dict[str, Any]) -> str:
    """Format architecture result as markdown for the CLI."""
    if "error" in result:
        return f"## Architecture Failed\n\n{result['error']}"

    lines = [
        f"## Architecture: {result['description']}",
        "",
        f"**Duration:** {result.get('duration', '0s')}",
        "",
        "---",
        "",
        result.get("architecture", "No architecture generated."),
    ]

    return "\n".join(lines)


# ── Private Helpers ──────────────────────────────────────────────────
# (No private helpers needed currently)
