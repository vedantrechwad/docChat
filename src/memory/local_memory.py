"""
Local Memory Layer — SQLite conversation history for DocChat.

Single-user, single-workspace memory. Stores conversation turns
and document metadata for the current session.
"""

import json
import sqlite3
import logging
from typing import Any, Dict, List
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalMemoryLayer:
    """
    SQLite-based memory for conversation history and source tracking.
    """

    def __init__(self, db_path: str = "./data/memory.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._setup_tables()
        logger.info(f"Memory initialized: {db_path}")

    def _setup_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                size TEXT DEFAULT '',
                chunk_count INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)

        self.conn.commit()

    def save_conversation_turn(self, rag_result) -> None:
        """Save a query + response pair."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute(
            "INSERT INTO conversations (role, content, sources_json, created_at) VALUES (?, ?, '[]', ?)",
            ("user", rag_result.query, now),
        )

        sources_json = json.dumps(rag_result.sources_used, default=str)
        cursor.execute(
            "INSERT INTO conversations (role, content, sources_json, created_at) VALUES (?, ?, ?, ?)",
            ("assistant", rag_result.response, sources_json, now),
        )

        self.conn.commit()

    def get_conversation_context(self, max_turns: int = 5) -> str:
        """Get recent conversation as formatted context string for prompts."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
            (max_turns * 2,),
        )
        rows = list(reversed(cursor.fetchall()))
        if not rows:
            return ""
        parts = []
        for row in rows:
            role = "User" if row["role"] == "user" else "Assistant"
            parts.append(f"{role}: {row['content']}")
        return "\n".join(parts)

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get full chat history for the frontend."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT role, content, sources_json, created_at FROM conversations ORDER BY id ASC")
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources_json"]),
            }
            for row in cursor.fetchall()
        ]

    def save_source(self, source_info: Dict[str, Any]) -> int:
        """Track an added source. Returns the source row ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO sources (name, source_type, size, chunk_count, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                source_info.get("name", "Unknown"),
                source_info.get("type", "unknown"),
                source_info.get("size", ""),
                source_info.get("chunks", 0),
                json.dumps(source_info),
                datetime.now().isoformat(),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_sources(self) -> List[Dict[str, Any]]:
        """Get all tracked sources."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, source_type, size, chunk_count, created_at FROM sources ORDER BY id ASC")
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "type": row["source_type"],
                "size": row["size"],
                "chunks": row["chunk_count"],
                "created_at": row["created_at"],
            }
            for row in cursor.fetchall()
        ]

    def delete_source(self, source_id: int) -> bool:
        """Remove a source by ID."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def clear_all(self) -> None:
        """Reset everything."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM conversations")
        cursor.execute("DELETE FROM sources")
        self.conn.commit()
        logger.info("Memory cleared")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
