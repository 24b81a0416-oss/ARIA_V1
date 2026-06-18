"""
ARIA — Persistent Memory

SQLite-backed persistent memory with FTS5 full-text search.
Stores conversations across sessions, user facts/preferences,
and provides context for the LLM on startup.

Tables:
  - conversations: Every chat message with role, content, timestamp
  - facts: Key-value user preferences and facts (e.g., "prefers React")
  - sessions: Tracking session start/end times

FTS5 virtual table enables full-text search across all conversation history.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re

from utils.vector_store import index_content

MEMORY_DIR = Path(__file__).parent.parent / ".aria"
DB_PATH = MEMORY_DIR / "memory.db"


# ── Database Initialization ──────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    """Get or create the database connection."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            start_time REAL NOT NULL,
            end_time REAL,
            summary TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            token_count INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            source TEXT DEFAULT 'chat',
            timestamp REAL NOT NULL,
            confidence REAL DEFAULT 1.0
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
        USING fts5(content, content=conversations, content_rowid=id);
    """)
    conn.commit()


# ── Session Management ───────────────────────────────────────────────

def start_session() -> str:
    """Start a new session and return its ID."""
    conn = get_db()
    session_id = str(uuid.uuid4())[:8]
    conn.execute(
        "INSERT INTO sessions (id, start_time) VALUES (?, ?)",
        (session_id, time.time()),
    )
    conn.commit()
    return session_id


def end_session(session_id: str, summary: str = "") -> None:
    """End a session."""
    conn = get_db()
    conn.execute(
        "UPDATE sessions SET end_time = ?, summary = ? WHERE id = ?",
        (time.time(), summary[:500], session_id),
    )
    conn.commit()


# ── Conversation Storage ─────────────────────────────────────────────

def save_message(
    session_id: str,
    role: str,
    content: str,
    token_count: int = 0,
) -> int:
    """Save a message to the conversation history. Auto-indexes to vector store."""
    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO conversations (session_id, role, content, timestamp, token_count)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, role, content, time.time(), token_count),
    )
    msg_id = cursor.lastrowid

    # Update FTS index
    try:
        conn.execute(
            "INSERT INTO conversations_fts(rowid, content) VALUES (?, ?)",
            (msg_id, content),
        )
    except sqlite3.OperationalError:
        pass  # FTS might not be available in all builds

    conn.commit()

    # Auto-index to vector store (async-capable: non-blocking)
    if len(content) >= 20:
        try:
            index_content(
                content,
                source="chat",
                metadata={"role": role, "session_id": session_id},
            )
        except Exception:
            pass  # Vector indexing is non-critical

    return msg_id


def get_recent_conversations(limit: int = 20) -> List[Dict[str, Any]]:
    """Get the most recent conversation messages."""
    conn = get_db()
    rows = conn.execute(
        """SELECT c.id, c.role, c.content, c.timestamp, c.session_id
           FROM conversations c
           ORDER BY c.timestamp DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()

    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"],
            "session_id": row["session_id"],
            "time_str": datetime.fromtimestamp(row["timestamp"]).strftime("%H:%M"),
        }
        for row in reversed(rows)  # Reverse to get chronological order
    ]


# ── Full-Text Search ─────────────────────────────────────────────────

def search_conversations(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Full-text search across all conversation history."""
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT c.id, c.role, c.content, c.timestamp, c.session_id,
                      snippet(conversations_fts, 0, '<<', '>>', '...', 40) AS highlighted
               FROM conversations_fts
               JOIN conversations c ON conversations_fts.rowid = c.id
               WHERE conversations_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # Fallback to LIKE search if FTS fails
        rows = conn.execute(
            """SELECT id, role, content, timestamp, session_id,
                      '' AS highlighted
               FROM conversations
               WHERE content LIKE ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (f"%{query}%", limit),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"][:300],
            "highlighted": row["highlighted"] if row["highlighted"] else "",
            "timestamp": row["timestamp"],
            "time_str": datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M"),
            "session_id": row["session_id"],
        }
        for row in rows
    ]


# ── Facts / User Preferences ─────────────────────────────────────────

def save_fact(key: str, value: str, category: str = "general", confidence: float = 1.0) -> None:
    """Save or update a fact about the user or project."""
    conn = get_db()
    conn.execute(
        """INSERT INTO facts (key, value, category, timestamp, confidence)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET
               value = excluded.value,
               timestamp = excluded.timestamp,
               confidence = excluded.confidence""",
        (key, value, category, time.time(), confidence),
    )
    conn.commit()


def get_facts(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all saved facts, optionally filtered by category."""
    conn = get_db()
    if category:
        rows = conn.execute(
            "SELECT * FROM facts WHERE category = ? ORDER BY timestamp DESC",
            (category,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM facts ORDER BY confidence DESC, timestamp DESC",
        ).fetchall()

    return [
        {
            "id": row["id"],
            "key": row["key"],
            "value": row["value"],
            "category": row["category"],
            "confidence": row["confidence"],
            "time_str": datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d"),
        }
        for row in rows
    ]


def delete_fact(key: str) -> bool:
    """Delete a fact by key. Returns True if deleted."""
    conn = get_db()
    cursor = conn.execute("DELETE FROM facts WHERE key = ?", (key,))
    conn.commit()
    return cursor.rowcount > 0


def get_relevant_context(current_input: str = "") -> Dict[str, Any]:
    """
    Get relevant context from memory to inject into the LLM.

    Returns:
        Dict with: recent_messages, relevant_facts, conversation_summary
    """
    facts = get_facts()
    recent = get_recent_conversations(20)

    # Search for relevant past conversations
    search_results = []
    if current_input:
        # Extract key terms for search
        terms = re.findall(r'\b\w{4,}\b', current_input.lower())
        for term in terms[:3]:
            results = search_conversations(term, limit=3)
            search_results.extend(results)

    # Format facts as context
    fact_lines = []
    for f in facts:
        fact_lines.append(f"- {f['key']}: {f['value']}")

    # Format recent messages
    recent_lines = []
    for msg in recent[-10:]:  # Last 10 messages
        role_label = "You" if msg["role"] == "user" else "ARIA"
        recent_lines.append(f"{role_label}: {msg['content'][:200]}")

    return {
        "facts": "\n".join(fact_lines) if fact_lines else "No saved facts yet.",
        "recent_conversations": "\n".join(recent_lines),
        "relevant_memories": "\n".join(
            f"[{s['time_str']}] {s['role']}: {s['content'][:200]}"
            for s in search_results[:5]
        ),
        "search_results_count": len(search_results),
    }


# ── Stats & Maintenance ──────────────────────────────────────────────

def get_stats() -> Dict[str, Any]:
    """Get memory usage statistics."""
    conn = get_db()
    conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    last_session = conn.execute(
        "SELECT start_time FROM sessions ORDER BY start_time DESC LIMIT 1"
    ).fetchone()

    # DB file size
    db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0

    return {
        "conversations": conv_count,
        "facts": fact_count,
        "sessions": session_count,
        "last_active": datetime.fromtimestamp(last_session[0]).strftime("%Y-%m-%d %H:%M")
        if last_session else "Never",
        "database_size": f"{db_size / 1024:.1f} KB",
        "database_path": str(DB_PATH),
    }


def forget_message(message_id: int) -> bool:
    """Delete a specific message from memory."""
    conn = get_db()
    cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (message_id,))
    try:
        conn.execute("DELETE FROM conversations_fts WHERE rowid = ?", (message_id,))
    except sqlite3.OperationalError:
        pass
    conn.commit()
    return cursor.rowcount > 0


def clear_all() -> None:
    """Clear all memory (conversations, facts, sessions)."""
    conn = get_db()
    conn.executescript("""
        DELETE FROM conversations_fts;
        DELETE FROM conversations;
        DELETE FROM facts;
        DELETE FROM sessions;
    """)
    conn.commit()
