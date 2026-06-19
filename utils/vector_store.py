"""
ARIA — Vector Store (Semantic Memory)

ChromaDB-backed vector store for semantic search.
Supports NVIDIA NIM embeddings (primary) with Ollama fallback.
No PyTorch, no HuggingFace downloads, no loading bars.

Features:
  - Auto-indexing of chat messages, research reports, rd reports
  - Semantic search (find by meaning, not just keywords)
  - Collection-based organization (chat, research, projects, facts)
  - Lightweight: ChromaDB + OpenAI-compatible API calls

Usage:
    from utils.vector_store import search_memory, index_content
    index_content("What is FastAPI?", source="chat")
    results = search_memory("Python web frameworks", limit=5)

Environment:
    NVIDIA_API_KEY       (required for NVIDIA NIM embeddings)
    NVIDIA_BASE_URL      (default: https://integrate.api.nvidia.com/v1)
    NVIDIA_EMBED_MODEL   (default: nvidia/nv-embed-v1)
    OLLAMA_BASE_URL      (fallback, default: http://localhost:11434)
    OLLAMA_EMBED_MODEL   (fallback, default: nomic-embed-text)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Configuration ────────────────────────────────────────────────────

_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
_NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
_NVIDIA_EMBED_MODEL = os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nv-embed-v1")

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

_VECTOR_STORE_DIR = Path(__file__).parent.parent / ".aria" / "vectors"
_STORE: Optional[Any] = None         # Singleton ChromaDB collection
_EMBEDDING_FN: Optional[Any] = None  # Singleton embedding function


# ── Embedding Providers ──────────────────────────────────────────────

class NvidiaEmbeddingFunction:
    """ChromaDB-compatible embedding function that calls NVIDIA NIM's /v1/embeddings.

    Uses the OpenAI-compatible NVIDIA API endpoint. Requires NVIDIA_API_KEY.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=self.base_url)

    def name(self) -> str:
        """ChromaDB calls embedding_function.name() to check for conflicts."""
        return f"nvidia_{self.model}"

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings via NVIDIA NIM API."""
        response = self._client.embeddings.create(
            model=self.model,
            input=input,
        )
        return [item.embedding for item in response.data]


class OllamaEmbeddingFunction:
    """ChromaDB-compatible embedding function that calls Ollama's /api/embed.

    Fallback provider when NVIDIA is not available.
    """

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def name(self) -> str:
        """ChromaDB calls embedding_function.name() to check for conflicts."""
        return f"ollama_{self.model}"

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings via Ollama API."""
        import requests

        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": input},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]


def _ping_ollama() -> bool:
    """Check if Ollama server is reachable."""
    try:
        import requests
        resp = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _get_embedding_function() -> Optional[Any]:
    """Get or create the verified embedding function singleton.

    Priority: NVIDIA NIM → Ollama (local) → None
    Verifies the provider works with a quick test call before caching.
    """
    global _EMBEDDING_FN
    if _EMBEDDING_FN is not None:
        return _EMBEDDING_FN

    # Try NVIDIA first: create + quick test call to verify endpoint works
    if _NVIDIA_API_KEY:
        try:
            fn = NvidiaEmbeddingFunction(_NVIDIA_API_KEY, _NVIDIA_BASE_URL, _NVIDIA_EMBED_MODEL)
            fn(["test"])  # Quick connectivity & model check
            _EMBEDDING_FN = fn
            return _EMBEDDING_FN
        except Exception:
            pass  # Fall through to Ollama

    # Fall back to Ollama
    if _ping_ollama():
        _EMBEDDING_FN = OllamaEmbeddingFunction(_OLLAMA_BASE_URL, _OLLAMA_EMBED_MODEL)
        return _EMBEDDING_FN

    return None  # No provider available


def _get_active_provider_name() -> str:
    """Return the name of the active embedding provider."""
    if _EMBEDDING_FN is not None:
        if isinstance(_EMBEDDING_FN, NvidiaEmbeddingFunction):
            return f"nvidia ({_NVIDIA_EMBED_MODEL})"
        elif isinstance(_EMBEDDING_FN, OllamaEmbeddingFunction):
            return f"ollama ({_OLLAMA_EMBED_MODEL})"
    return "none"


def _get_store():
    """Get or create the ChromaDB collection singleton."""
    global _STORE
    if _STORE is not None:
        return _STORE

    embedding_fn = _get_embedding_function()
    if embedding_fn is None:
        return None  # Embeddings not available

    try:
        import chromadb
        _VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(str(_VECTOR_STORE_DIR))
        collection = client.get_or_create_collection(
            name="aria_memory",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        _STORE = collection
        return _STORE
    except ImportError:
        return None
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────

def is_available() -> bool:
    """Check if the vector store is available (NVIDIA or Ollama + ChromaDB)."""
    try:
        import chromadb
        # Try to get the embedding function (tests actual connectivity)
        fn = _get_embedding_function()
        return fn is not None
    except ImportError:
        return False


def index_content(
    content: str,
    source: str = "chat",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Index a piece of content into the vector store for semantic search.

    Args:
        content: The text to index
        source: Category (chat, research, rd, project, fact)
        metadata: Optional dict with extra info (role, topic, etc.)

    Returns:
        True if indexed, False if skipped/failed
    """
    if not content or len(content.strip()) < 20:
        return False  # Skip very short content

    # Truncate very long content to avoid embedding provider token limits
    # Most models have max ~512 tokens (~2000 chars)
    MAX_CHARS = 2000
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS]

    store = _get_store()
    if store is None:
        return False  # Not available

    try:
        doc_id = f"{source}_{int(time.time() * 1000)}_{hash(content) % 10**9}"

        meta: Dict[str, Any] = {
            "source": source,
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M"),
        }
        if metadata:
            meta.update(metadata)

        store.add(
            documents=[content],
            metadatas=[meta],
            ids=[doc_id],
        )
        return True
    except Exception:
        return False


def search_memory(
    query: str,
    limit: int = 5,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search across all indexed content.

    Args:
        query: The search query (natural language)
        limit: Max results to return
        source_filter: Optional category filter (chat, research, rd, etc.)

    Returns:
        List of dicts with: content, source, score, time_str, metadata
    """
    if not query or len(query.strip()) < 2:
        return []

    store = _get_store()
    if store is None:
        return []

    try:
        where = None
        if source_filter:
            where = {"source": source_filter}

        results = store.query(
            query_texts=[query],
            n_results=min(limit, 20),
            where=where,
        )

        output = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                meta = {}
                if results.get("metadatas") and len(results["metadatas"][0]) > i:
                    meta = results["metadatas"][0][i]

                score = None
                if results.get("distances") and len(results["distances"][0]) > i:
                    # Convert cosine distance to similarity score (0-1)
                    score = max(0.0, 1.0 - results["distances"][0][i])

                output.append({
                    "content": doc[:500],
                    "source": meta.get("source", "unknown"),
                    "score": score,
                    "time_str": meta.get("time_str", ""),
                    "metadata": meta,
                })

        return output
    except Exception:
        return []


def get_stats() -> Dict[str, Any]:
    """Get vector store statistics."""
    store = _get_store()
    if store is None:
        return {
            "available": False,
            "count": 0,
            "error": "No embedding provider available",
            "nvidia_key": bool(_NVIDIA_API_KEY),
            "ollama_running": _ping_ollama() if not _NVIDIA_API_KEY else False,
        }

    try:
        count = store.count()
        return {
            "available": True,
            "count": count,
            "path": str(_VECTOR_STORE_DIR),
            "provider": _get_active_provider_name(),
        }
    except Exception as e:
        return {"available": True, "count": -1, "error": str(e)}


def clear_all() -> bool:
    """Clear all vectors from the store."""
    global _STORE
    try:
        import chromadb
        _VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(str(_VECTOR_STORE_DIR))
        try:
            client.delete_collection("aria_memory")
        except Exception:
            pass
        _STORE = None  # Reset singleton
        return True
    except Exception:
        return False
