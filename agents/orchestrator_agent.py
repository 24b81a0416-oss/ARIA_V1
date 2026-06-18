"""
ARIA — Orchestrator Agent

Multi-agent task decomposition and orchestration.
Takes complex tasks, breaks them into sub-tasks, executes focused
sub-agents for each, and synthesizes the results.

This is how ARIA handles complex multi-step problems that benefit from
a plan → execute → review → refine workflow.

Commands:
  `orchestrate [task]`  — Decompose and execute a complex task
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from utils.llm import LLMClient
from utils.skill_manager import get_instructions_for_task


def run_orchestration(
    task: str,
    llm: LLMClient,
    llm_deep: Optional[LLMClient] = None,
    user_context: str = "",
) -> Dict[str, Any]:
    """
    Run multi-agent orchestration for a complex task.

    Pipeline:
      1. Decompose task into sub-tasks
      2. Execute each sub-task (sequentially)
      3. Synthesize all sub-results into final output

    Args:
        task: The complex task to execute
        llm: Primary LLM (fast, for decomposition and execution)
        llm_deep: Optional deeper LLM for complex sub-tasks (e.g., NVIDIA)

    Returns:
        Dict with: task, sub_tasks[], results[], final_output, duration
    """
    start = time.time()
    print(f"  [Orchestrator] Task: {task}")
    print(f"  [Orchestrator] Phase 1: Decomposing task...")

    # Load relevant skills
    skill_context, skill_names = get_instructions_for_task(task, llm)
    if skill_names:
        print(f"  [Orchestrator] Loaded skills: {', '.join(skill_names)}")

    # ── Phase 1: Decompose ───────────────────────────────────────────
    decompose_prompt = f"""Decompose this task into 2-4 focused sub-tasks:

Task: {task}

{skill_context}

For each sub-task, provide:
1. A clear, specific description
2. What information or output it should produce
3. Whether it needs the deep reasoning model (NVIDIA) or the fast model (Groq)

Output format - one sub-task per line:
ST: <description> | <output expectation> | <model: fast/deep>

Be specific. Each sub-task should be independently executable.
"""
    decompose_system = "You are a senior engineering manager. Decompose complex tasks into independently executable sub-tasks."
    if user_context:
        decompose_system += f"\n\n### User Context\n{user_context}"

    try:
        decomposition = llm.generate(
            decompose_prompt,
            system_prompt=decompose_system,
            max_tokens=1024,
        )
    except Exception as e:
        return {"error": f"Decomposition failed: {e}", "task": task}

    # ── Parse sub-tasks ──────────────────────────────────────────────
    sub_tasks = _parse_sub_tasks(decomposition)
    if not sub_tasks:
        # Fallback: treat the whole task as one sub-task
        sub_tasks = [{"description": task, "output": "Complete solution", "model": "fast"}]

    print(f"  [Orchestrator] Created {len(sub_tasks)} sub-tasks")

    # ── Phase 2: Execute ─────────────────────────────────────────────
    results = []
    for i, st in enumerate(sub_tasks, 1):
        desc = st["description"]
        model_type = st.get("model", "fast")
        client_to_use = llm_deep if (model_type == "deep" and llm_deep) else llm

        print(f"  [Sub-task {i}/{len(sub_tasks)}] {desc[:60]}...")

        # Build context from previous results
        context = ""
        if results:
            context = "\n\nPrevious sub-task results:\n"
            for j, prev in enumerate(results, 1):
                context += f"\n--- Result {j}: {prev.get('description', '')} ---\n{prev.get('output', '')[:1000]}\n"

        exec_prompt = f"""Execute this sub-task:

Sub-task: {desc}
Expected output: {st.get('output', 'Complete solution')}
{context}

Provide a clear, complete result.
"""

        exec_system = "You are a focused, specialized agent executing a single sub-task. Be complete but concise."
        if user_context:
            exec_system += f"\n\n### User Context\n{user_context}"

        try:
            output = client_to_use.generate(
                exec_prompt,
                system_prompt=exec_system,
                max_tokens=2048,
            )
            results.append({
                "description": desc,
                "output": output.strip(),
                "model": model_type,
            })
            print(f"    ✓ Complete ({len(output)} chars)")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results.append({
                "description": desc,
                "output": f"[Error: {e}]",
                "model": model_type,
                "error": str(e),
            })

    # ── Phase 3: Synthesize ──────────────────────────────────────────
    print(f"  [Orchestrator] Phase 3: Synthesizing results...")

    synthesis_prompt = f"""Synthesize these sub-task results into a coherent final output.

Original task: {task}

Sub-task results:
"""
    for i, r in enumerate(results, 1):
        synthesis_prompt += f"""
--- Sub-task {i}: {r['description']} ---
{r['output'][:1500]}
"""

    synthesis_prompt += """
Create a cohesive, well-structured final output that combines all sub-task results.
Include:
1. Executive summary of what was accomplished
2. Detailed results organized logically
3. Any dependencies or relationships between parts

Format in clear markdown.
"""

    synthesis_system = "You are a senior technical writer. Synthesize multiple results into a cohesive, well-structured document."
    if user_context:
        synthesis_system += f"\n\n### User Context\n{user_context}"

    try:
        final_output = llm.generate(
            synthesis_prompt,
            system_prompt=synthesis_system,
            max_tokens=3072,
        )
    except Exception as e:
        final_output = f"Synthesis failed: {e}"

    duration = time.time() - start
    print(f"  [Orchestrator] Complete ({duration:.0f}s) — {len(sub_tasks)} sub-tasks")

    return {
        "task": task,
        "sub_tasks": [{"description": st["description"], "model": st.get("model", "fast")}
                      for st in sub_tasks],
        "results": results,
        "final_output": final_output.strip(),
        "skill_names": skill_names,
        "duration": f"{duration:.0f}s",
    }


def format_orchestration_result(result: Dict[str, Any]) -> str:
    """Format orchestration result as markdown."""
    if "error" in result:
        return f"## Orchestration Failed\n\n{result['error']}"

    lines = [
        f"## Orchestration: {result['task']}",
        "",
        f"**Sub-tasks:** {len(result.get('sub_tasks', []))}",
        f"**Duration:** {result.get('duration', '0s')}",
        "",
    ]

    skills = result.get("skill_names", [])
    if skills:
        lines.append(f"**Skills used:** {', '.join(skills)}")
        lines.append("")

    lines.extend([
        "### Sub-tasks",
        "",
    ])
    for i, st in enumerate(result.get("sub_tasks", []), 1):
        icon = "✅" if i <= len(result.get("results", [])) else "⏳"
        model_icon = "🧠" if st.get("model") == "deep" else "⚡"
        lines.append(f"{icon} {model_icon} {st['description']}")

    lines.extend([
        "",
        "---",
        "",
        result.get("final_output", "No output generated."),
    ])

    return "\n".join(lines)


# ── Private Helpers ──────────────────────────────────────────────────

def _parse_sub_tasks(decomposition: str) -> List[Dict[str, str]]:
    """Parse LLM decomposition output into structured sub-tasks."""
    sub_tasks = []
    for line in decomposition.split("\n"):
        line = line.strip()
        if line.startswith("ST:"):
            content = line[3:].strip()
            parts = [p.strip() for p in content.split("|", 2)]
            if len(parts) >= 1:
                sub_task = {"description": parts[0], "output": "", "model": "fast"}
                if len(parts) >= 2:
                    sub_task["output"] = parts[1]
                if len(parts) >= 3:
                    model = parts[2].strip().lower()
                    sub_task["model"] = "deep" if model in ("deep", "nvidia") else "fast"
                sub_tasks.append(sub_task)
    return sub_tasks
