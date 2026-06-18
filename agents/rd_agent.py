"""
ARIA — R&D Agent

Deep research capabilities:
  1. `rd [topic]`           — Multi-angle deep research with synthesis
  2. `rd compare A vs B`    — Technology comparison with feature matrix
  3. `rd feasibility [idea]` — Feasibility analysis
  4. `rd competitive [market]` — Competitive landscape analysis

Uses Groq for search + synthesis (fast, cheap) and NVIDIA for deep analysis (optional).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from utils.researcher import search_web, extract_content
from utils.llm import LLMClient


# ── Public API ─────────────────────────────────────────────────────────

def run_rd_research(
    topic: str,
    llm_client: LLMClient,
    mode: str = "deep",  # "deep", "compare", "feasibility", "competitive"
    max_sources: int = 8,
    user_context: str = "",
) -> Dict[str, Any]:
    """
    Run R&D research with the given mode.

    Args:
        topic: The research topic / question
        llm_client: LLM for synthesis and analysis
        mode: Research mode
            "deep"       — multi-angle deep research
            "compare"    — technology comparison
            "feasibility" — feasibility analysis
            "competitive" — competitive landscape
        max_sources: Max sources to fetch per query

    Returns:
        Dict with keys: topic, mode, report, sources, duration
    """
    start = time.time()

    modes = {
        "deep": _do_deep_research,
        "compare": _do_comparison,
        "feasibility": _do_feasibility,
        "competitive": _do_competitive,
    }

    handler = modes.get(mode, _do_deep_research)

    print(f"  [R&D] Mode: {mode} — {topic}")

    try:
        result = handler(topic, llm_client, max_sources, user_context)
        result["mode"] = mode
        result["duration"] = f"{time.time() - start:.0f}s"
        print(f"  [R&D] Complete ({result['duration']}) — {result.get('source_count', 0)} sources")
        return result
    except Exception as e:
        duration = time.time() - start
        return {
            "topic": topic,
            "mode": mode,
            "report": f"## R&D Error\n\nResearch failed: {e}",
            "sources": [],
            "source_count": 0,
            "duration": f"{duration:.0f}s",
        }


def format_rd_report(result: Dict[str, Any]) -> str:
    """Format R&D result as markdown for CLI display."""
    mode = result.get("mode", "deep")
    topic = result.get("topic", "Unknown")
    report = result.get("report", "No report generated.")
    sources = result.get("sources", [])

    mode_labels = {
        "deep": "Deep Research",
        "compare": "Technology Comparison",
        "feasibility": "Feasibility Analysis",
        "competitive": "Competitive Analysis",
    }
    mode_label = mode_labels.get(mode, "Research")

    lines = [
        f"## {mode_label}: {topic}",
        "",
        f"**Sources:** {result.get('source_count', 0)} | **Duration:** {result.get('duration', '0s')}",
        "",
        "---",
        "",
        report,
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


# ── Private: Deep Research ────────────────────────────────────────────

def _do_deep_research(
    topic: str,
    llm: LLMClient,
    max_sources: int,
    user_context: str = "",
) -> Dict[str, Any]:
    """Multi-angle deep research — generates sub-queries, searches each, synthesizes."""
    print(f"  [R&D] Generating research angles...")

    # Generate sub-queries from different angles
    query_prompt = f"""Research topic: {topic}

Generate 3 specific search queries to cover different angles of this topic.
Each query should be a focused question or search term that would return useful results.

Format: One query per line, no numbering.
"""
    query_system = "Generate specific search queries. One per line."
    if user_context:
        query_system += f"\n\n### User Context\n{user_context}"

    try:
        queries_raw = llm.generate(
            query_prompt,
            system_prompt=query_system,
            max_tokens=256,
        )
        queries = [q.strip() for q in queries_raw.strip().split("\n") if q.strip()[:3] != "---" and q.strip()]
        # Clean up markdown or numbering
        queries = [re.sub(r"^\d+[\.\)]\s*", "", q).strip('"').strip("'").strip() for q in queries]
        queries = [q for q in queries if len(q) > 5][:5]
    except Exception:
        queries = [topic]

    # If the topic itself isn't included, add it
    if topic not in queries:
        queries.insert(0, topic)

    print(f"  [R&D] Searching {len(queries)} angles: {', '.join(q[:40] for q in queries)}")

    # Search each query, collect unique results
    all_results = []
    seen_urls = set()
    failed_queries = 0
    for q in queries:
        try:
            q_results = search_web(q, max_results=4)
            if not q_results:
                failed_queries += 1
                continue
            for r in q_results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
        except Exception:
            failed_queries += 1
            continue

    if not all_results:
        return _fallback_result(topic, "No search results found.")

    if failed_queries > 0:
        print(f"  [R&D] {failed_queries} queries returned no results")
    print(f"  [R&D] Found {len(all_results)} unique results across {len(queries) - failed_queries}/{len(queries)} queries")
    print(f"  [R&D] Fetching top content...")

    # Fetch content from top results
    sources = []
    skipped = 0
    for r in all_results[:max_sources]:
        content = extract_content(r["url"], max_chars=3000)
        if content and len(content.strip()) > 100:
            sources.append({
                "title": r.get("title", "Untitled"),
                "url": r["url"],
                "snippet": r.get("snippet", ""),
                "content": content,
                "source_index": len(sources) + 1,
            })

    if not sources:
        sources = [
            {"title": r.get("title", "Untitled"), "url": r["url"],
             "snippet": r.get("snippet", ""), "content": r.get("snippet", "")[:1000],
             "source_index": i + 1}
            for i, r in enumerate(all_results[:max_sources])
        ]

    print(f"  [R&D] Synthesizing {len(sources)} sources...")

    # Build context with inline citation markers
    context = ""
    for s in sources:
        idx = s["source_index"]
        context += f"\n--- Source {idx}: {s['title']} ---\nURL: {s['url']}\n{s['content'][:2000]}\n"

    synthesis_prompt = f"""Research topic: {topic}

Below are {len(sources)} web sources researched from {len(queries)} different angles.

{context}

Generate a comprehensive deep research report with:

1. **Executive Summary** — Concise overview of key findings (2-3 sentences)
2. **Key Insights** — 4-6 substantive findings with supporting evidence
3. **Analysis** — Deeper analysis of implications, trends, and connections
4. **Conclusion** — Summary and recommended next steps

**IMPORTANT - Cite your sources inline:** When referencing information from a source, add the source number in brackets, e.g. "Vector databases excel at similarity search [Source 1]..." This lets readers trace claims back to the original source.

Format in clear markdown. Be specific and cite evidence from the sources.
"""

    deep_system = "You are a senior research analyst. Produce detailed, evidence-backed research reports."
    if user_context:
        deep_system += f"\n\n### User Context\n{user_context}"

    try:
        report = llm.generate(
            synthesis_prompt,
            system_prompt=deep_system,
            max_tokens=3072,
        )
    except Exception as e:
        report = f"Report generation failed: {e}"

    return {
        "topic": topic,
        "report": report.strip(),
        "sources": [s["url"] for s in sources],
        "source_count": len(sources),
    }


# ── Private: Technology Comparison ────────────────────────────────────

def _do_comparison(
    topic: str,
    llm: LLMClient,
    max_sources: int,
    user_context: str = "",
) -> Dict[str, Any]:
    """Compare two or more technologies side-by-side."""
    # Parse "A vs B" format
    parts = re.split(r"\s+vs\.?\s+", topic, maxsplit=1)
    if len(parts) >= 2:
        tech_a = parts[0].strip()
        tech_b = parts[1].strip()
    else:
        tech_a = topic
        tech_b = ""

    print(f"  [R&D] Comparing: {tech_a} vs {tech_b or '(general comparison)'}")

    # Build queries
    queries = [f"{tech_a} features pros cons"]
    if tech_b:
        queries.append(f"{tech_b} features pros cons")
        queries.append(f"{tech_a} vs {tech_b} comparison")
    else:
        queries.append(f"alternatives to {tech_a} comparison")

    # Search
    all_results = []
    seen_urls = set()
    for q in queries:
        results = search_web(q, max_results=4)
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    if not all_results:
        return _fallback_result(topic, "No search results found for comparison.")

    # Fetch content
    sources = []
    skipped = 0
    for r in all_results[:max_sources]:
        content = extract_content(r["url"], max_chars=2500)
        if content and len(content.strip()) > 100:
            sources.append({
                "title": r.get("title", "Untitled"),
                "url": r["url"],
                "content": content,
                "source_index": len(sources) + 1,
            })
        elif content:
            skipped += 1

    if not sources:
        sources = [{"title": r.get("title", "Untitled"), "url": r["url"],
                     "content": r.get("snippet", "")[:1000], "source_index": i + 1}
                    for i, r in enumerate(all_results[:max_sources])]

    context = ""
    for s in sources:
        idx = s["source_index"]
        context += f"\n--- Source {idx}: {s['title']} ---\nURL: {s['url']}\n{s['content'][:2000]}\n"

    compare_prompt = f"""Compare these technologies based on the research below.

Tech A: {tech_a}
{f'Tech B: {tech_b}' if tech_b else 'Alternatives'}

{context}

Generate a structured comparison report with:

1. **Overview** — Brief intro to each technology
2. **Feature Comparison Table** — Markdown table with columns: Feature | {tech_a} | {tech_b or 'Alternatives'}
3. **Pros & Cons** — For each technology
4. **Use Case Fit** — Which scenario suits each technology best
5. **Recommendation** — With reasoning

**IMPORTANT - Cite your sources inline:** When referencing information, add the source number in brackets, e.g. "[Source 1]".

Be objective. Use the sources for evidence.
"""

    compare_system = "You are a technology analyst. Produce structured, objective comparisons with tables."
    if user_context:
        compare_system += f"\n\n### User Context\n{user_context}"

    try:
        report = llm.generate(
            compare_prompt,
            system_prompt=compare_system,
            max_tokens=3072,
        )
    except Exception as e:
        report = f"Comparison failed: {e}"

    return {
        "topic": topic,
        "report": report.strip(),
        "sources": [{"url": s["url"], "title": s["title"]} for s in sources],
        "source_count": len(sources),
    }


# ── Private: Feasibility Analysis ─────────────────────────────────────

def _do_feasibility(
    topic: str,
    llm: LLMClient,
    max_sources: int,
    user_context: str = "",
) -> Dict[str, Any]:
    """Analyze the feasibility of an idea or project."""
    queries = [
        topic,
        f"{topic} implementation challenges",
        f"{topic} best practices guide",
        f"{topic} examples case study",
    ]

    all_results = []
    seen_urls = set()
    for q in queries:
        results = search_web(q, max_results=3)
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    if not all_results:
        return _fallback_result(topic, "No search results found for feasibility analysis.")

    sources = []
    skipped = 0
    for r in all_results[:max_sources]:
        content = extract_content(r["url"], max_chars=2500)
        if content and len(content.strip()) > 100:
            sources.append({"title": r.get("title", "Untitled"), "url": r["url"],
                           "snippet": r.get("snippet", ""), "content": content,
                           "source_index": len(sources) + 1})
        elif content:
            skipped += 1

    if not sources:
        sources = [{"title": r.get("title", "Untitled"), "url": r["url"],
                     "snippet": r.get("snippet", ""), "content": r.get("snippet", "")[:1000],
                     "source_index": i + 1}
                    for i, r in enumerate(all_results[:max_sources])]

    context = ""
    for s in sources:
        idx = s["source_index"]
        context += f"\n--- Source {idx}: {s['title']} ---\nURL: {s['url']}\n{s['content'][:2000]}\n"

    fease_prompt = f"""Feasibility analysis for: {topic}

{context}

Generate a structured feasibility report covering:

1. **Executive Summary** — Is this feasible? What's the verdict?
2. **Technical Feasibility** — What technologies/tools are needed? Maturity?
3. **Resource Requirements** — Estimated effort, skills, infrastructure
4. **Risk Assessment** — Key risks, challenges, mitigation strategies
5. **Timeline Estimate** — Rough phases and milestones
6. **Recommendation** — Go / No-Go with reasoning

**IMPORTANT - Cite your sources inline:** Add the source number in brackets, e.g. "[Source 1]".

Be realistic and balanced. Identify showstoppers if any.
"""

    fease_system = "You are a senior technical project manager. Assess feasibility realistically."
    if user_context:
        fease_system += f"\n\n### User Context\n{user_context}"

    try:
        report = llm.generate(
            fease_prompt,
            system_prompt=fease_system,
            max_tokens=3072,
        )
    except Exception as e:
        report = f"Feasibility analysis failed: {e}"

    return {
        "topic": topic,
        "report": report.strip(),
        "sources": [s["url"] for s in sources],
        "source_count": len(sources),
    }


# ── Private: Competitive Analysis ─────────────────────────────────────

def _do_competitive(
    topic: str,
    llm: LLMClient,
    max_sources: int,
    user_context: str = "",
) -> Dict[str, Any]:
    """Analyze the competitive landscape for a market or domain."""
    queries = [
        f"{topic} market landscape 2025 2026",
        f"{topic} top companies competitors",
        f"{topic} market share trends",
        f"{topic} industry analysis",
    ]

    all_results = []
    seen_urls = set()
    for q in queries:
        results = search_web(q, max_results=3)
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    if not all_results:
        return _fallback_result(topic, "No results found for competitive analysis.")

    sources = []
    skipped = 0
    for r in all_results[:max_sources]:
        content = extract_content(r["url"], max_chars=2500)
        if content and len(content.strip()) > 100:
            sources.append({"title": r.get("title", "Untitled"), "url": r["url"],
                           "content": content,
                           "source_index": len(sources) + 1})
        elif content:
            skipped += 1

    if not sources:
        sources = [{"title": r.get("title", "Untitled"), "url": r["url"],
                     "content": r.get("snippet", "")[:1000], "source_index": i + 1}
                    for i, r in enumerate(all_results[:max_sources])]

    context = ""
    for s in sources:
        idx = s["source_index"]
        context += f"\n--- Source {idx}: {s['title']} ---\nURL: {s['url']}\n{s['content'][:2000]}\n"

    comp_prompt = f"""Competitive landscape analysis for: {topic}

{context}

Generate a structured competitive analysis report:

1. **Market Overview** — Size, growth, key trends
2. **Key Players** — Who are the main competitors? Their positioning?
3. **Competitive Comparison** — Features, pricing, strengths, weaknesses
4. **Market Gaps** — Unmet needs and opportunities
5. **Strategic Insights** — Implications and recommended positioning
6. **SWOT Summary** — Strengths, Weaknesses, Opportunities, Threats

**IMPORTANT - Cite your sources inline:** Add the source number in brackets, e.g. "[Source 1]".

Include specific company names and details from the sources.
"""

    comp_system = "You are a competitive intelligence analyst. Produce data-driven market analysis."
    if user_context:
        comp_system += f"\n\n### User Context\n{user_context}"

    try:
        report = llm.generate(
            comp_prompt,
            system_prompt=comp_system,
            max_tokens=3072,
        )
    except Exception as e:
        report = f"Competitive analysis failed: {e}"

    return {
        "topic": topic,
        "report": report.strip(),
        "sources": [{"url": s["url"], "title": s["title"]} for s in sources],
        "source_count": len(sources),
    }


# ── Shared Helper ─────────────────────────────────────────────────────

def _fallback_result(topic: str, reason: str) -> Dict[str, Any]:
    """Return a minimal fallback result."""
    return {
        "topic": topic,
        "sources": [],
        "source_count": 0,
        "report": f"## No Results\n\n{reason}",
    }
