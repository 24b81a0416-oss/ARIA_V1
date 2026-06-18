"""
ARIA — Researcher Module

Web search and content extraction for the Research Agent.
Uses DuckDuckGo for search (free, no API key) and requests + BeautifulSoup for content extraction.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import requests
from bs4 import BeautifulSoup


# ─── Search ─────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 8) -> List[dict]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        List of dicts with 'title', 'url', 'snippet'
    """
    # Try multiple DDGS import paths (package was renamed)
    ddgs_class = None
    for module_path in ["ddgs", "duckduckgo_search"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mod = __import__(module_path, fromlist=["DDGS"])
            ddgs_class = getattr(mod, "DDGS", None)
            if ddgs_class:
                break
        except ImportError:
            continue

    if ddgs_class:
        try:
            with ddgs_class() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
                return results
        except Exception as e:
            print(f"  [Research] DDGS search failed: {e}")

    # Fallback: try a basic HTTP request to DuckDuckGo's HTML API
    return _fallback_search(query, max_results)


def _fallback_search(query: str, max_results: int = 8) -> List[dict]:
    """Fallback search using DuckDuckGo's HTML API."""
    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        results = []
        for result in soup.select(".result")[:max_results]:
            title_el = result.select_one(".result__title a")
            snippet_el = result.select_one(".result__snippet")

            if title_el:
                results.append({
                    "title": title_el.get_text(strip=True),
                    "url": title_el.get("href", ""),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })
        return results
    except Exception as e:
        print(f"  [Research] Fallback search failed: {e}")
        return []


# ─── Content Extraction ─────────────────────────────────────────────────────

def extract_content(url: str, max_chars: int = 5000) -> Optional[str]:
    """
    Fetch a URL and extract readable text content.

    Args:
        url: The URL to fetch
        max_chars: Maximum characters to extract

    Returns:
        Extracted text content, or None if failed
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract text from main content areas first
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        # Truncate
        if len(text) > max_chars:
            text = text[:max_chars] + "\n..."

        return text

    except Exception as e:
        print(f"  [Research] Failed to fetch {url}: {e}")
        return None
