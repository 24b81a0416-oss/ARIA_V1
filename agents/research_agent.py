"""
ARIA — Research Agent

Orchestrates the full research pipeline:
  1. Search the web for the topic
  2. Fetch content from top results
  3. Synthesize findings using the LLM
  4. Generate a structured research report

Uses Groq for synthesis (fast, good for text) and NVIDIA optionally for deeper analysis.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from utils.researcher import search_web, extract_content
from utils.llm import LLMClient


def run_research(
    topic: str,
    llm_client: LLMClient,
    max_sources: int = 5,
    user_context: str = "",
) -> Dict[str, Any]:
    """
    Run the full research pipeline.

    Args:
        topic: What to research
        llm_client: LLM client for synthesis
        max_sources: Max sources to fetch and analyze

    Returns:
        Dict with keys: topic, summary, findings, sources, duration, report
    """
    start = time.time()
    print(f"  [Research] Searching for: {topic}")

    # ── Step 1: Search ─────────────────────────────────────────────────
    results = search_web(topic, max_results=max_sources + 3)
    if not results:
        return {
            "topic": topic,
            "summary": "No search results found.",
            "findings": "",
            "sources": [],
            "duration": f"{time.time() - start:.0f}s",
            "report": f"No search results found for: {topic}",
        }

    print(f"  [Research] Found {len(results)} results, fetching content...")

    # ── Step 2: Fetch content from top results ─────────────────────────
    sources = []
    skipped = 0
    for r in results[:max_sources]:
        content = extract_content(r["url"], max_chars=3000)
        if content and len(content.strip()) > 100:
            sources.append({
                "title": r["title"] or "Untitled",
                "url": r["url"],
                "snippet": r.get("snippet", ""),
                "content": content,
                "source_index": len(sources) + 1,
            })
            print(f"    [Fetch {len(sources)}] {r['title'][:60] if r['title'] else 'Untitled'}...")
        elif content:
            skipped += 1

    if not sources:
        print(f"  [Research] No substantial content fetched ({skipped} too short), using snippets...")
        sources = [
            {"title": r.get("title", "Untitled"), "url": r["url"],
             "snippet": r.get("snippet", ""), "content": r.get("snippet", "")[:1000],
             "source_index": i + 1}
            for i, r in enumerate(results[:max_sources])
        ]
    elif skipped > 0:
        print(f"  [Research] Skipped {skipped} low-content results")

    print(f"  [Research] Synthesizing {len(sources)} sources...")

    # ── Step 3: Build context for LLM with inline citation markers ─────
    sources_text = ""
    for s in sources:
        idx = s["source_index"]
        sources_text += f"\n--- Source {idx}: {s['title']} ---\nURL: {s['url']}\n{s['content'][:2000]}\n"

    synthesis_prompt = f"""Research topic: {topic}

Below are {len(sources)} web sources about this topic. Synthesize them into a clear, structured research report.

{sources_text}

Generate a structured report with:
1. **Executive Summary** - 2-3 sentence overview of findings
2. **Key Findings** - 3-5 main points with supporting details
3. **Analysis** - Deeper context and connections between findings

**IMPORTANT - Cite your sources inline:** When referencing information from a source, add the source number in brackets, e.g. "Python is widely used for AI [Source 1]..." This lets the reader trace claims back to the original source.

Format using markdown."""

    # ── Step 4: Generate report ────────────────────────────────────────
    print(f"  [Research] Generating report...")
    base_system = "You are a research analyst. Synthesize information from multiple sources into clear, accurate, structured reports. Be objective and cite key facts."
    if user_context:
        base_system += f"\n\n## About the user\n{user_context}"

    try:
        response = llm_client.generate(
            synthesis_prompt,
            system_prompt=base_system,
            max_tokens=2048,
        )
        report = response.strip()
    except Exception as e:
        report = f"Report generation failed: {e}"

    duration = time.time() - start

    # ── Step 5: Build result ───────────────────────────────────────────
    result = {
        "topic": topic,
        "sources": [{"url": s["url"], "title": s["title"]} for s in sources],
        "duration": f"{duration:.0f}s",
        "report": report,
        "source_count": len(sources),
    }

    print(f"  [Research] Complete ({duration:.0f}s) — {len(sources)} sources analyzed")
    return result


def format_report(result: Dict[str, Any]) -> str:
    """Format research result as a clean markdown summary for the CLI."""
    report_text = result.get("report", "No report generated.")
    sources = result.get("sources", [])

    lines = [
        f"## Research: {result.get('topic', 'Unknown')}",
        "",
        f"**Sources analyzed:** {result.get('source_count', 0)}",
        f"**Duration:** {result.get('duration', '0s')}",
        "",
        "---",
        "",
        report_text,
        "",
    ]
    if sources:
        lines.append("---")
        lines.append("")
        lines.append("**Sources:**")
        for s in sources[:10]:
            title = s.get("title", "") if isinstance(s, dict) else ""
            url = s.get("url", s) if isinstance(s, dict) else s
            if title:
                lines.append(f"- [{title}]({url})")
            else:
                lines.append(f"- {url}")

    return "\n".join(lines)
