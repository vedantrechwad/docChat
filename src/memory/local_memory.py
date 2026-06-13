"""
Local Memory Layer - SQLite-based conversation memory.

Replaces the paid Zep Cloud API with a fully local SQLite database.
Stores conversation history, user preferences, and document metadata.
"""

import logging
import json
import sqlite3
from typing import Any, Dict, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single conversation turn with context."""
    user_query: str
    assistant_response: str
    sources_used: List[Dict[str, Any]]
    timestamp: str
    session_id: str


class LocalMemoryLayer:
    """
    Local SQLite-based memory layer that replaces Zep Cloud.

    Provides:
    - Conversation history storage and retrieval
    - User preferences persistence
    - Document metadata tracking
    - Session management
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        db_path: str = "./data/memory.db",
        create_new_session: bool = False,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.db_path = db_path

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._setup_tables()

        if create_new_session:
            self._clear_session_messages()

        logger.info(f"LocalMemoryLayer initialized for user={user_id}, session={session_id}")

    def _setup_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT DEFAULT '[]',
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, key)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                doc_name TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                doc_size TEXT,
                chunk_count INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
        """)

        # Index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session 
            ON conversations(session_id, user_id)
        """)

        self.conn.commit()

    def save_conversation_turn(self, rag_result) -> None:
        """
        Save a conversation turn (user query + assistant response).

        Args:
            rag_result: A RAGResult object with query, response, and sources_used.
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        # Save user message
        cursor.execute("""
            INSERT INTO conversations (session_id, user_id, role, content, sources_json, metadata_json, created_at)
            VALUES (?, ?, 'user', ?, '[]', '{}', ?)
        """, (self.session_id, self.user_id, rag_result.query, now))

        # Save assistant response
        sources_json = json.dumps(rag_result.sources_used, default=str)
        metadata = {
            "retrieval_count": rag_result.retrieval_count,
            "sources_count": len(rag_result.sources_used),
        }
        cursor.execute("""
            INSERT INTO conversations (session_id, user_id, role, content, sources_json, metadata_json, created_at)
            VALUES (?, ?, 'assistant', ?, ?, ?, ?)
        """, (
            self.session_id, self.user_id,
            rag_result.response, sources_json,
            json.dumps(metadata), now
        ))

        self.conn.commit()
        logger.info(f"Saved conversation turn with {len(rag_result.sources_used)} sources")

    def get_conversation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent conversation history for the current session."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, content, sources_json, metadata_json, created_at
            FROM conversations
            WHERE session_id = ? AND user_id = ?
            ORDER BY id ASC
            LIMIT ?
        """, (self.session_id, self.user_id, limit))

        rows = cursor.fetchall()
        history = []
        for row in rows:
            entry = {
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources_json"]),
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            history.append(entry)

        return history

    def get_conversation_context(self, max_turns: int = 10) -> str:
        """
        Get a formatted conversation context string for use in prompts.
        Returns the last N turns as a formatted string.
        """
        history = self.get_conversation_history(limit=max_turns * 2)
        if not history:
            return ""

        context_parts = []
        for entry in history:
            role = "User" if entry["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {entry['content']}")

        return "\n".join(context_parts)

    def get_relevant_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Simple keyword-based memory search.
        For a more sophisticated approach, we could add embeddings to memory entries.
        """
        cursor = self.conn.cursor()
        
        # Simple LIKE-based search across all sessions for this user
        keywords = query.lower().split()[:5]  # Take top 5 keywords
        conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        params.extend([self.user_id, limit])

        cursor.execute(f"""
            SELECT role, content, sources_json, session_id, created_at
            FROM conversations
            WHERE ({conditions}) AND user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, params)

        rows = cursor.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources_json"]),
                "session_id": row["session_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """Save or update user preferences."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        for key, value in preferences.items():
            cursor.execute("""
                INSERT INTO user_preferences (user_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET value = ?, updated_at = ?
            """, (self.user_id, key, json.dumps(value), now, json.dumps(value), now))

        self.conn.commit()
        logger.info(f"Saved {len(preferences)} user preferences")

    def get_user_preferences(self) -> Dict[str, Any]:
        """Retrieve all user preferences."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT key, value FROM user_preferences WHERE user_id = ?
        """, (self.user_id,))

        prefs = {}
        for row in cursor.fetchall():
            try:
                prefs[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                prefs[row["key"]] = row["value"]

        return prefs

    def save_document_metadata(self, document_info: Dict[str, Any]) -> None:
        """Track a processed document."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO document_metadata (session_id, user_id, doc_name, doc_type, doc_size, chunk_count, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.session_id, self.user_id,
            document_info.get("name", "Unknown"),
            document_info.get("type", "unknown"),
            document_info.get("size", ""),
            document_info.get("chunks", 0),
            json.dumps(document_info),
            datetime.now().isoformat(),
        ))
        self.conn.commit()
        logger.info(f"Saved document metadata: {document_info.get('name')}")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) as user_count,
                   SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) as assistant_count,
                   MIN(created_at) as first_msg,
                   MAX(created_at) as last_msg
            FROM conversations
            WHERE session_id = ? AND user_id = ?
        """, (self.session_id, self.user_id))

        row = cursor.fetchone()
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "total_messages": row["total"] or 0,
            "user_messages": row["user_count"] or 0,
            "assistant_messages": row["assistant_count"] or 0,
            "first_message": row["first_msg"],
            "last_message": row["last_msg"],
        }

    def clear_session(self) -> None:
        """Clear all messages in the current session."""
        self._clear_session_messages()
        logger.info(f"Session {self.session_id} cleared")

    def _clear_session_messages(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM conversations WHERE session_id = ? AND user_id = ?
        """, (self.session_id, self.user_id))
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Memory database connection closed")
