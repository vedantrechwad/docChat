"""
Local Memory Layer — SQLite storage for notebooks, conversations, sources, and notes.

Multi-notebook, single-user architecture. Each notebook has its own
chat history, sources, and notes.
"""

import json
import sqlite3
import logging
import threading
import os
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
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
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
                updated_at TEXT NOT NULL,
                is_private INTEGER DEFAULT 0,
                password_hash TEXT DEFAULT NULL
            )
        """)

        try:
            cursor.execute("ALTER TABLE notebooks ADD COLUMN concepts_generated INTEGER DEFAULT 0")
        except Exception:
            pass

        try:
            cursor.execute("ALTER TABLE notebooks ADD COLUMN is_private INTEGER DEFAULT 0")
        except Exception:
            pass

        try:
            cursor.execute("ALTER TABLE notebooks ADD COLUMN password_hash TEXT DEFAULT NULL")
        except Exception:
            pass

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
                indexed_in_rag INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS source_files (
                source_id INTEGER PRIMARY KEY,
                notebook_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                mime_type TEXT DEFAULT '',
                checksum TEXT DEFAULT '',
                revision INTEGER DEFAULT 1,
                page_text_json TEXT DEFAULT '{}',
                FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS source_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                revision INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notebook_id INTEGER NOT NULL UNIQUE,
                title TEXT DEFAULT 'Untitled',
                html_content TEXT DEFAULT '',
                updated_at TEXT NOT NULL,
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notebook_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                explanation TEXT NOT NULL,
                links_json TEXT DEFAULT '[]',
                x INTEGER DEFAULT 100,
                y INTEGER DEFAULT 100,
                sort_order INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            )
        """)

        # Self-healing migration for existing concepts table
        try:
            cursor.execute("ALTER TABLE concepts ADD COLUMN x INTEGER DEFAULT 100")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE concepts ADD COLUMN y INTEGER DEFAULT 100")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE concepts ADD COLUMN sort_order INTEGER DEFAULT 0")
        except Exception:
            pass

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_text (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notebook_id INTEGER NOT NULL,
                source_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                citation_json TEXT DEFAULT '{}',
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE,
                FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_text_notebook
            ON chunk_text(notebook_id)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS study_guides (
                notebook_id INTEGER PRIMARY KEY,
                content_json TEXT,
                updated_at TEXT,
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            )
        """)

        self.conn.commit()
        self._migrate_notes_column()
        self._ensure_default_performance_mode()

    def _migrate_notes_column(self):
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(notes)")
        cols = [row[1] for row in cursor.fetchall()]
        if "indexed_in_rag" not in cols:
            cursor.execute("ALTER TABLE notes ADD COLUMN indexed_in_rag INTEGER DEFAULT 0")
            self.conn.commit()

    def _ensure_default_performance_mode(self):
        if not self.get_setting("performance_mode"):
            self.set_setting("performance_mode", "fast")
        if not self.get_setting("ingest_mode"):
            self.set_setting("ingest_mode", "fast")

    def _ensure_default_notebook(self):
        """Create a 'Default' notebook if none exist."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM notebooks")
        if cursor.fetchone()[0] == 0:
            self.create_notebook("Default")

    # ─── Notebooks ─────────────────────────────────────────────────────────

    def create_notebook(self, name: str, is_private: int = 0, password_hash: Optional[str] = None) -> int:
        """Create a new notebook. Returns its ID."""
        with self._lock:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO notebooks (name, created_at, updated_at, is_private, password_hash) VALUES (?, ?, ?, ?, ?)",
                (name, now, now, is_private, password_hash),
            )
            self.conn.commit()
            return cursor.lastrowid

    def list_notebooks(self) -> List[Dict[str, Any]]:
        """List all notebooks."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, created_at, updated_at, is_private, password_hash FROM notebooks ORDER BY updated_at DESC")
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
                "is_private": row["is_private"] or 0,
                "password_hash": row["password_hash"],
            })
        return notebooks

    def verify_notebook_password(self, notebook_id: int, password_hash: str) -> bool:
        """Verify password hash for private notebook."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT password_hash FROM notebooks WHERE id = ?", (notebook_id,))
        row = cursor.fetchone()
        if not row:
            return False
        stored_hash = row[0]
        if not stored_hash:
            return True
        return stored_hash == password_hash

    def rename_notebook(self, notebook_id: int, name: str) -> bool:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE notebooks SET name = ?, updated_at = ? WHERE id = ?",
                (name, datetime.now().isoformat(), notebook_id),
            )
            self.conn.commit()
            return cursor.rowcount > 0

    def delete_notebook(self, notebook_id: int) -> bool:
        """Delete a notebook and all its data."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
            self.conn.commit()
            return cursor.rowcount > 0

    def _touch_notebook(self, notebook_id: int):
        """Update the notebook's updated_at timestamp. Must be called within self._lock."""
        self.conn.execute(
            "UPDATE notebooks SET updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(), notebook_id),
        )

    # ─── Conversations ─────────────────────────────────────────────────────

    def save_conversation_turn(self, rag_result, notebook_id: int = 1) -> None:
        """Save a query + response pair."""
        with self._lock:
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

    def get_conversation_context(self, notebook_id: int = 1, max_turns: int = 5, max_chars: int = 4000) -> str:
        """Get recent conversation as formatted context string.
        
        Ensures only complete Q&A pairs are included and total size
        stays within max_chars to avoid blowing the LLM context window.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content FROM conversations WHERE notebook_id = ? ORDER BY id DESC LIMIT ?",
            (notebook_id, max_turns * 2),
        )
        rows = list(reversed(cursor.fetchall()))
        if not rows:
            return ""
        # Ensure we have complete pairs (user + assistant)
        paired_rows = []
        i = 0
        while i < len(rows) - 1:
            if rows[i]["role"] == "user" and rows[i + 1]["role"] == "assistant":
                paired_rows.extend([rows[i], rows[i + 1]])
                i += 2
            else:
                i += 1  # Skip orphaned messages
        parts = []
        total_chars = 0
        for row in paired_rows:
            role = "User" if row["role"] == "user" else "Assistant"
            line = f"{role}: {row['content']}"
            if total_chars + len(line) > max_chars and parts:
                break
            parts.append(line)
            total_chars += len(line)
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
        with self._lock:
            self.conn.execute("DELETE FROM conversations WHERE notebook_id = ?", (notebook_id,))
            self.conn.commit()

    # ─── Sources ───────────────────────────────────────────────────────────

    def save_source(self, source_info: Dict[str, Any], notebook_id: int = 1) -> int:
        """Track an added source. Returns the source row ID."""
        with self._lock:
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
            "SELECT id, name, source_type, size, chunk_count, metadata_json, created_at FROM sources WHERE notebook_id = ? ORDER BY id ASC",
            (notebook_id,),
        )
        rows = []
        for row in cursor.fetchall():
            meta = json.loads(row["metadata_json"] or "{}")
            rows.append({
                "id": row["id"],
                "name": row["name"],
                "display_name": meta.get("title") or row["name"],
                "type": row["source_type"],
                "size": row["size"],
                "chunks": row["chunk_count"],
                "created_at": row["created_at"],
                "index_status": meta.get("index_status", "ready"),
            })
        return rows

    def get_source_by_id(self, source_id: int) -> Optional[Dict[str, Any]]:
        """Get a source record by ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, notebook_id, name, source_type, metadata_json FROM sources WHERE id = ?",
            (source_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "notebook_id": row["notebook_id"],
            "name": row["name"],
            "type": row["source_type"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    def get_source_metadata(self, source_id: int) -> Dict[str, Any]:
        """Get the full metadata JSON for a source."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT metadata_json FROM sources WHERE id = ?", (source_id,))
        row = cursor.fetchone()
        if not row:
            return {}
        return json.loads(row["metadata_json"] or "{}")

    def update_source(self, source_id: int, source_info: Dict[str, Any], notebook_id: Optional[int] = None) -> bool:
        """Update an existing source in-place (for refresh). Preserves the source ID."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE sources SET name = ?, size = ?, chunk_count = ?, metadata_json = ? WHERE id = ?",
                (
                    source_info.get("name", "Unknown"),
                    source_info.get("size", ""),
                    source_info.get("chunks", 0),
                    json.dumps(source_info),
                    source_id,
                ),
            )
            self.conn.commit()
            if notebook_id is not None:
                self._touch_notebook(notebook_id)
            return cursor.rowcount > 0

    def delete_source(self, source_id: int) -> bool:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM chunk_text WHERE source_id = ?", (source_id,))
            cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            self.conn.commit()
            return cursor.rowcount > 0

    # ─── Chunk text (BM25 corpus) ─────────────────────────────────────────

    def save_chunk_texts(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self._lock:
            self.conn.executemany(
                """INSERT INTO chunk_text (notebook_id, source_id, chunk_index, content, citation_json)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        r["notebook_id"],
                        r["source_id"],
                        r["chunk_index"],
                        r["content"],
                        json.dumps(r.get("citation_json", {})),
                    )
                    for r in rows
                ],
            )
            self.conn.commit()

    def get_chunk_texts_by_notebook(self, notebook_id: int) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT id, notebook_id, source_id, chunk_index, content, citation_json
               FROM chunk_text WHERE notebook_id = ? ORDER BY source_id, chunk_index""",
            (notebook_id,),
        )
        docs = []
        for row in cursor.fetchall():
            try:
                citation = json.loads(row["citation_json"] or "{}")
            except json.JSONDecodeError:
                citation = {}
            docs.append({
                "id": row["id"],
                "content": row["content"],
                "citation": citation,
                "metadata": {},
                "embedding_model": "",
            })
        return docs

    def delete_chunk_texts_by_source(self, source_id: int) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM chunk_text WHERE source_id = ?", (source_id,))
            self.conn.commit()

    def delete_chunk_texts_by_notebook(self, notebook_id: int) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM chunk_text WHERE notebook_id = ?", (notebook_id,))
            self.conn.commit()

    # ─── Notes ─────────────────────────────────────────────────────────────

    def create_note(self, notebook_id: int, title: str, content: str = "") -> int:
        """Create a new note. Returns its ID."""
        with self._lock:
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
        with self._lock:
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
        cursor.execute("SELECT id, notebook_id, title, content, indexed_in_rag, created_at, updated_at FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "notebook_id": row["notebook_id"],
            "title": row["title"],
            "content": row["content"],
            "indexed_in_rag": bool(row["indexed_in_rag"]) if "indexed_in_rag" in row.keys() else False,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def delete_note(self, note_id: int) -> bool:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            self.conn.commit()
            return cursor.rowcount > 0

    def set_note_indexed(self, note_id: int, indexed: bool) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE notes SET indexed_in_rag = ?, updated_at = ? WHERE id = ?",
                (1 if indexed else 0, datetime.now().isoformat(), note_id),
            )
            self.conn.commit()

    # ─── Settings ──────────────────────────────────────────────────────────

    def get_setting(self, key: str, default: str = "") -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self.conn.commit()

    def get_performance_mode(self) -> str:
        mode = self.get_setting("performance_mode", "fast")
        return mode if mode in ("fast", "quality") else "fast"

    def set_performance_mode(self, mode: str) -> None:
        if mode not in ("fast", "quality"):
            mode = "fast"
        self.set_setting("performance_mode", mode)

    def get_chunking_settings(self, notebook_id: int = 1) -> Dict[str, Any]:
        raw = self.get_setting(f"chunking_notebook_{notebook_id}", "")
        perf = self.get_performance_mode()
        default_ingest = "fast" if perf == "fast" else "quality"
        if raw:
            try:
                settings = json.loads(raw)
                if perf == "fast":
                    settings["ingest_mode"] = "fast"
                return settings
            except json.JSONDecodeError:
                pass
        return {
            "preset": self.get_setting("chunking_preset", "auto"),
            "chunk_tokens": int(self.get_setting("chunk_tokens", "384") or 384),
            "overlap_tokens": int(self.get_setting("overlap_tokens", "100") or 100),
            "ingest_mode": self.get_setting("ingest_mode", default_ingest),
        }

    def set_chunking_settings(self, settings: Dict[str, Any], notebook_id: int = 1) -> None:
        self.set_setting(f"chunking_notebook_{notebook_id}", json.dumps(settings))

    def get_discover_settings(self, notebook_id: int = 1) -> Dict[str, Any]:
        raw = self.get_setting(f"discover_notebook_{notebook_id}", "")
        defaults = {
            "enabled": False,
            "auto_on_topic": False,
            "max_results": 8,
            "provider": "duckduckgo",
            "api_key": os.getenv("DISCOVER_API_KEY", ""),
        }
        if raw:
            try:
                defaults.update(json.loads(raw))
            except json.JSONDecodeError:
                pass
        return defaults

    def set_discover_settings(self, settings: Dict[str, Any], notebook_id: int = 1) -> None:
        self.set_setting(f"discover_notebook_{notebook_id}", json.dumps(settings))

    # ─── Source files ──────────────────────────────────────────────────────

    def save_source_file(
        self,
        source_id: int,
        notebook_id: int,
        file_path: str,
        mime_type: str = "",
        checksum: str = "",
        page_text: Optional[Dict[str, str]] = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO source_files (source_id, notebook_id, file_path, mime_type, checksum, page_text_json)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_id) DO UPDATE SET
                   file_path=excluded.file_path, mime_type=excluded.mime_type,
                   checksum=excluded.checksum, page_text_json=excluded.page_text_json,
                   revision=revision+1""",
                (source_id, notebook_id, file_path, mime_type, checksum, json.dumps(page_text or {})),
            )
            self.conn.commit()

    def get_source_file(self, source_id: int) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT source_id, notebook_id, file_path, mime_type, checksum, revision, page_text_json FROM source_files WHERE source_id = ?",
            (source_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "source_id": row[0],
            "notebook_id": row[1],
            "file_path": row[2],
            "mime_type": row[3],
            "checksum": row[4],
            "revision": row[5],
            "page_text": json.loads(row[6] or "{}"),
        }

    def save_source_revision(self, source_id: int, revision: int, content: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO source_revisions (source_id, revision, content, created_at) VALUES (?, ?, ?, ?)",
                (source_id, revision, content, datetime.now().isoformat()),
            )
            self.conn.commit()

    def find_source_by_name(self, name: str, notebook_id: int) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, notebook_id, name, source_type FROM sources WHERE name = ? AND notebook_id = ?",
            (name, notebook_id),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {"id": row[0], "notebook_id": row[1], "name": row[2], "type": row[3]}

    def find_source_by_checksum(self, checksum: str, notebook_id: int) -> Optional[Dict[str, Any]]:
        if not checksum:
            return None
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT s.id, s.name, s.source_type, s.chunk_count, sf.checksum
               FROM source_files sf
               JOIN sources s ON s.id = sf.source_id
               WHERE sf.checksum = ? AND sf.notebook_id = ?""",
            (checksum, notebook_id),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "type": row[2],
            "chunks": row[3],
            "checksum": row[4],
            "notebook_id": notebook_id,
        }

    # ─── Documents (Quill) ─────────────────────────────────────────────────

    def get_document(self, notebook_id: int) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, notebook_id, title, html_content, updated_at FROM documents WHERE notebook_id = ?",
            (notebook_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "notebook_id": row["notebook_id"],
            "title": row["title"],
            "html_content": row["html_content"],
            "updated_at": row["updated_at"],
        }

    def save_document(self, notebook_id: int, html_content: str, title: str = "Untitled") -> int:
        with self._lock:
            now = datetime.now().isoformat()
            self.conn.execute(
                """INSERT INTO documents (notebook_id, title, html_content, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(notebook_id) DO UPDATE SET
                   title=excluded.title, html_content=excluded.html_content, updated_at=excluded.updated_at""",
                (notebook_id, title, html_content, now),
            )
            self.conn.commit()
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM documents WHERE notebook_id = ?", (notebook_id,))
            row = cursor.fetchone()
            return row[0] if row else 0

    def save_study_guide(self, notebook_id: int, content_json: str) -> None:
        """Save or update study guide for a notebook."""
        from datetime import datetime
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO study_guides (notebook_id, content_json, updated_at)
               VALUES (?, ?, ?)""",
            (notebook_id, content_json, datetime.now().isoformat())
        )
        self.conn.commit()

    def get_study_guide(self, notebook_id: int) -> Optional[str]:
        """Get study guide for a notebook."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT content_json FROM study_guides WHERE notebook_id = ?",
            (notebook_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

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

    def list_concepts(self, notebook_id: int) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, notebook_id, title, explanation, links_json, x, y, sort_order, created_at FROM concepts WHERE notebook_id = ? ORDER BY sort_order ASC, id ASC",
            (notebook_id,)
        )
        concepts = []
        for r in cursor.fetchall():
            concepts.append({
                "id": r[0],
                "notebook_id": r[1],
                "title": r[2],
                "explanation": r[3],
                "links": json.loads(r[4] or "[]"),
                "x": r[5] if r[5] is not None else 100,
                "y": r[6] if r[6] is not None else 100,
                "sort_order": r[7] if r[7] is not None else 0,
                "created_at": r[8]
            })
        return concepts

    def create_concept(self, notebook_id: int, title: str, explanation: str, links_json: str = "[]", x: int = 100, y: int = 100, sort_order: int = 0) -> int:
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO concepts (notebook_id, title, explanation, links_json, x, y, sort_order, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (notebook_id, title, explanation, links_json, x, y, sort_order, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_concept(self, concept_id: int, title: Optional[str] = None, explanation: Optional[str] = None, links_json: Optional[str] = None, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        cursor = self.conn.cursor()
        updates = []
        params = []
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if explanation is not None:
            updates.append("explanation = ?")
            params.append(explanation)
        if links_json is not None:
            updates.append("links_json = ?")
            params.append(links_json)
        if x is not None:
            updates.append("x = ?")
            params.append(x)
        if y is not None:
            updates.append("y = ?")
            params.append(y)
        if not updates:
            return False
        params.append(concept_id)
        cursor.execute(
            f"UPDATE concepts SET {', '.join(updates)} WHERE id = ?",
            tuple(params)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def shift_concepts_order(self, notebook_id: int):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE concepts SET sort_order = sort_order + 1 WHERE notebook_id = ?", (notebook_id,))
        self.conn.commit()

    def update_concept_order(self, concept_id: int, sort_order: int):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE concepts SET sort_order = ? WHERE id = ?", (sort_order, concept_id))
        self.conn.commit()

    def delete_concept(self, concept_id: int) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM concepts WHERE id = ?", (concept_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def has_generated_concepts(self, notebook_id: int) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT concepts_generated FROM notebooks WHERE id = ?", (notebook_id,))
        row = cursor.fetchone()
        return bool(row and row[0])

    def mark_concepts_generated(self, notebook_id: int):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE notebooks SET concepts_generated = 1 WHERE id = ?", (notebook_id,))
        self.conn.commit()

    def grade_concept_card(self, concept_id: int, grade: str) -> int:
        """Update flashcard sort_order (Leitner Box level 1-5) based on user recall grade."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT sort_order FROM concepts WHERE id = ?", (concept_id,))
        row = cursor.fetchone()
        current_box = row[0] if row else 1
        if not current_box or current_box < 1:
            current_box = 1

        if grade == "easy":
            new_box = min(5, current_box + 1)
        elif grade == "hard":
            new_box = 1
        else:  # good
            new_box = current_box

        cursor.execute("UPDATE concepts SET sort_order = ? WHERE id = ?", (new_box, concept_id))
        self.conn.commit()
        return new_box

    def close(self) -> None:
        if self.conn:
            self.conn.close()
