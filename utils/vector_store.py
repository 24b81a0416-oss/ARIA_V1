"""
ARIA — Vector Store (Semantic Memory)

ChromaDB-backed vector store for semantic search across conversations,
research reports, generated projects, and facts.

Features:
  - Auto-indexing of chat messages, research reports, rd reports
  - Semantic search (find by meaning, not just keywords)
  - Collection-based organization (chat, research, projects, facts)
  - Lightweight: ChromaDB runs locally, no external services needed

Usage:
    from utils.vector_store import get_vector_store, search_memory, index_content
    vs = get_vector_store()
    index_content("chat", "What is FastAPI?", "user")
    results = search_memory("Python web frameworks", limit=5)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Lazy imports (only loaded when vector store is actually used) ─────

_VECTOR_STORE_DIR = Path(__file__).parent.parent / ".aria" / "vectors"
_STORE: Optional[Any] = None   # Singleton ChromaDB collection
_EMBEDDING_FN: Optional[Any] = None  # Singleton embedding function


def _get_embedding_function():
    """Get or create the embedding function (lazy-loaded)."""
    global _EMBEDDING_FN
    if _EMBEDDING_FN is not None:
        return _EMBEDDING_FN

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBEDDING_FN = model.encode
        return _EMBEDDING_FN
    except ImportError:
        return None
    except Exception:
        return None


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
    """Check if the vector store is available (deps installed)."""
    try:
        import chromadb
        import sentence_transformers
        return True
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

    store = _get_store()
    if store is None:
        return False  # Not available

    try:
        # Generate a stable-ish ID from content hash + timestamp
        doc_id = f"{source}_{int(time.time() * 1000)}_{hash(content) % 10**9}"

        meta = {
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
        return {"available": False, "count": 0, "error": "Not available"}

    try:
        count = store.count()
        return {
            "available": True,
            "count": count,
            "path": str(_VECTOR_STORE_DIR),
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
