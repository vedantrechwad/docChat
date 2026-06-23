"""
Local Memory Layer — SQLite storage for notebooks, conversations, sources, and notes.

Multi-notebook, single-user architecture. Each notebook has its own
chat history, sources, and notes.
"""

import json
import sqlite3
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalMemoryLayer:
    """
    SQLite-based memory with multi-notebook support.

    Schema:
        notebooks: id, name, created_at, updated_at
        conversations: id, notebook_id, role, content, sources_json, created_at
        sources: id, notebook_id, name, source_type, size, chunk_count, metadata_json, created_at
        notes: id, notebook_id, title, content, created_at, updated_at
    """

    def __init__(self, db_path: str = "./data/memory.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._setup_tables()
        self._ensure_default_notebook()
        logger.info(f"Memory initialized: {db_path}")

    def _setup_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notebooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notebook_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notebook_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                size TEXT DEFAULT '',
                chunk_count INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notebook_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            )
        """)

        self.conn.commit()

    def _ensure_default_notebook(self):
        """Create a 'Default' notebook if none exist."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM notebooks")
        if cursor.fetchone()[0] == 0:
            self.create_notebook("Default")

    # ─── Notebooks ─────────────────────────────────────────────────────────

    def create_notebook(self, name: str) -> int:
        """Create a new notebook. Returns its ID."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO notebooks (name, created_at, updated_at) VALUES (?, ?, ?)",
            (name, now, now),
        )
        self.conn.commit()
        return cursor.lastrowid

    def list_notebooks(self) -> List[Dict[str, Any]]:
        """List all notebooks."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, created_at, updated_at FROM notebooks ORDER BY updated_at DESC")
        notebooks = []
        for row in cursor.fetchall():
            nb_id = row["id"]
            # Get counts
            cursor2 = self.conn.cursor()
            cursor2.execute("SELECT COUNT(*) FROM sources WHERE notebook_id = ?", (nb_id,))
            source_count = cursor2.fetchone()[0]
            cursor2.execute("SELECT COUNT(*) FROM conversations WHERE notebook_id = ? AND role = 'user'", (nb_id,))
            chat_count = cursor2.fetchone()[0]
            cursor2.execute("SELECT COUNT(*) FROM notes WHERE notebook_id = ?", (nb_id,))
            note_count = cursor2.fetchone()[0]

            notebooks.append({
                "id": nb_id,
                "name": row["name"],
                "sources": source_count,
                "chats": chat_count,
                "notes": note_count,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return notebooks

    def rename_notebook(self, notebook_id: int, name: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE notebooks SET name = ?, updated_at = ? WHERE id = ?",
            (name, datetime.now().isoformat(), notebook_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_notebook(self, notebook_id: int) -> bool:
        """Delete a notebook and all its data."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def _touch_notebook(self, notebook_id: int):
        """Update the notebook's updated_at timestamp."""
        self.conn.execute(
            "UPDATE notebooks SET updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(), notebook_id),
        )

    # ─── Conversations ─────────────────────────────────────────────────────

    def save_conversation_turn(self, rag_result, notebook_id: int = 1) -> None:
        """Save a query + response pair."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute(
            "INSERT INTO conversations (notebook_id, role, content, sources_json, created_at) VALUES (?, ?, ?, '[]', ?)",
            (notebook_id, "user", rag_result.query, now),
        )

        sources_json = json.dumps(rag_result.sources_used, default=str)
        cursor.execute(
            "INSERT INTO conversations (notebook_id, role, content, sources_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (notebook_id, "assistant", rag_result.response, sources_json, now),
        )

        self.conn.commit()
        self._touch_notebook(notebook_id)

    def get_conversation_context(self, notebook_id: int = 1, max_turns: int = 5) -> str:
        """Get recent conversation as formatted context string."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content FROM conversations WHERE notebook_id = ? ORDER BY id DESC LIMIT ?",
            (notebook_id, max_turns * 2),
        )
        rows = list(reversed(cursor.fetchall()))
        if not rows:
            return ""
        parts = []
        for row in rows:
            role = "User" if row["role"] == "user" else "Assistant"
            parts.append(f"{role}: {row['content']}")
        return "\n".join(parts)

    def get_chat_history(self, notebook_id: int = 1) -> List[Dict[str, Any]]:
        """Get full chat history for a notebook."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, role, content, sources_json, created_at FROM conversations WHERE notebook_id = ? ORDER BY id ASC",
            (notebook_id,),
        )
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources_json"]),
                "created_at": row["created_at"],
            }
            for row in cursor.fetchall()
        ]

    def clear_chat(self, notebook_id: int) -> None:
        """Clear chat history for a notebook."""
        self.conn.execute("DELETE FROM conversations WHERE notebook_id = ?", (notebook_id,))
        self.conn.commit()

    # ─── Sources ───────────────────────────────────────────────────────────

    def save_source(self, source_info: Dict[str, Any], notebook_id: int = 1) -> int:
        """Track an added source. Returns the source row ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO sources (notebook_id, name, source_type, size, chunk_count, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                notebook_id,
                source_info.get("name", "Unknown"),
                source_info.get("type", "unknown"),
                source_info.get("size", ""),
                source_info.get("chunks", 0),
                json.dumps(source_info),
                datetime.now().isoformat(),
            ),
        )
        self.conn.commit()
        self._touch_notebook(notebook_id)
        return cursor.lastrowid

    def get_sources(self, notebook_id: int = 1) -> List[Dict[str, Any]]:
        """Get all tracked sources for a notebook."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, name, source_type, size, chunk_count, created_at FROM sources WHERE notebook_id = ? ORDER BY id ASC",
            (notebook_id,),
        )
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
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ─── Notes ─────────────────────────────────────────────────────────────

    def create_note(self, notebook_id: int, title: str, content: str = "") -> int:
        """Create a new note. Returns its ID."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO notes (notebook_id, title, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (notebook_id, title, content, now, now),
        )
        self.conn.commit()
        self._touch_notebook(notebook_id)
        return cursor.lastrowid

    def update_note(self, note_id: int, title: Optional[str] = None, content: Optional[str] = None) -> bool:
        """Update a note's title and/or content."""
        cursor = self.conn.cursor()
        updates = []
        params = []
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if not updates:
            return False
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(note_id)

        cursor.execute(f"UPDATE notes SET {', '.join(updates)} WHERE id = ?", params)
        self.conn.commit()
        return cursor.rowcount > 0

    def append_to_note(self, note_id: int, text: str) -> bool:
        """Append text to an existing note."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT content FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if not row:
            return False
        new_content = row["content"] + "\n\n" + text if row["content"] else text
        return self.update_note(note_id, content=new_content)

    def list_notes(self, notebook_id: int) -> List[Dict[str, Any]]:
        """List all notes in a notebook."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, title, content, created_at, updated_at FROM notes WHERE notebook_id = ? ORDER BY updated_at DESC",
            (notebook_id,),
        )
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "preview": (row["content"] or "")[:100],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in cursor.fetchall()
        ]

    def get_note(self, note_id: int) -> Optional[Dict[str, Any]]:
        """Get a single note by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, notebook_id, title, content, created_at, updated_at FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "notebook_id": row["notebook_id"],
            "title": row["title"],
            "content": row["content"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def delete_note(self, note_id: int) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ─── Cleanup ───────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Reset everything."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM notes")
        cursor.execute("DELETE FROM conversations")
        cursor.execute("DELETE FROM sources")
        cursor.execute("DELETE FROM notebooks")
        self.conn.commit()
        self._ensure_default_notebook()
        logger.info("Memory cleared")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
