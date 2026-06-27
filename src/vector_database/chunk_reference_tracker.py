"""Chunk reference tracker - manages many-to-many chunk-to-notebook relationships."""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Set, Optional

logger = logging.getLogger(__name__)


class ChunkReferenceTracker:
    """Tracks which notebooks reference which chunks for proper deletion."""
    
    def __init__(self, db_path: str = "./data/chunk_references.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for chunk references."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_references (
                    chunk_id TEXT NOT NULL,
                    notebook_id INTEGER NOT NULL,
                    source_file TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, notebook_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunk_references(chunk_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notebook_id ON chunk_references(notebook_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source_file ON chunk_references(source_file)")
            conn.commit()
    
    def add_reference(self, chunk_id: str, notebook_id: int, source_file: str) -> bool:
        """Add a notebook reference to a chunk."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """INSERT OR IGNORE INTO chunk_references 
                           (chunk_id, notebook_id, source_file)
                           VALUES (?, ?, ?)""",
                        (chunk_id, notebook_id, source_file)
                    )
                    conn.commit()
                return True
            except Exception as e:
                logger.warning(f"Failed to add chunk reference: {e}")
                return False
    
    def add_references_batch(self, chunk_ids: list, notebook_id: int, source_file: str) -> bool:
        """Add multiple notebook references to chunks."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    for chunk_id in chunk_ids:
                        conn.execute(
                            """INSERT OR IGNORE INTO chunk_references 
                               (chunk_id, notebook_id, source_file)
                               VALUES (?, ?, ?)""",
                            (chunk_id, notebook_id, source_file)
                        )
                    conn.commit()
                logger.info(f"Added {len(chunk_ids)} chunk references for notebook {notebook_id}")
                return True
            except Exception as e:
                logger.warning(f"Failed to add batch chunk references: {e}")
                return False
    
    def remove_reference(self, chunk_id: str, notebook_id: int) -> bool:
        """Remove a notebook reference from a chunk."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM chunk_references WHERE chunk_id = ? AND notebook_id = ?",
                        (chunk_id, notebook_id)
                    )
                    conn.commit()
                    return cursor.rowcount > 0
            except Exception as e:
                logger.warning(f"Failed to remove chunk reference: {e}")
                return False
    
    def get_reference_count(self, chunk_id: str) -> int:
        """Get number of notebooks referencing a chunk."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM chunk_references WHERE chunk_id = ?",
                        (chunk_id,)
                    )
                    row = cursor.fetchone()
                    return row[0] if row else 0
            except Exception as e:
                logger.warning(f"Failed to get reference count: {e}")
                return 0
    
    def get_notebooks_for_chunk(self, chunk_id: str) -> Set[int]:
        """Get all notebook IDs that reference a chunk."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT notebook_id FROM chunk_references WHERE chunk_id = ?",
                        (chunk_id,)
                    )
                    return {row[0] for row in cursor.fetchall()}
            except Exception as e:
                logger.warning(f"Failed to get notebooks for chunk: {e}")
                return set()
    
    def get_chunks_for_source(self, source_file: str, notebook_id: Optional[int] = None) -> Set[str]:
        """Get all chunk IDs for a source file (optionally filtered by notebook)."""
        with self._lock:
            try:
                if notebook_id is not None:
                    cursor = sqlite3.connect(self.db_path).execute(
                        "SELECT chunk_id FROM chunk_references WHERE source_file = ? AND notebook_id = ?",
                        (source_file, notebook_id)
                    )
                else:
                    cursor = sqlite3.connect(self.db_path).execute(
                        "SELECT chunk_id FROM chunk_references WHERE source_file = ?",
                        (source_file,)
                    )
                return {row[0] for row in cursor.fetchall()}
            except Exception as e:
                logger.warning(f"Failed to get chunks for source: {e}")
                return set()
    
    def get_chunks_for_notebook(self, notebook_id: int) -> Set[str]:
        """Get all chunk IDs referenced by a notebook."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT chunk_id FROM chunk_references WHERE notebook_id = ?",
                        (notebook_id,)
                    )
                    return {row[0] for row in cursor.fetchall()}
            except Exception as e:
                logger.warning(f"Failed to get chunks for notebook: {e}")
                return set()
    
    def delete_by_source(self, source_file: str, notebook_id: Optional[int] = None) -> int:
        """Delete all references for a source file."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    if notebook_id is not None:
                        cursor = conn.execute(
                            "DELETE FROM chunk_references WHERE source_file = ? AND notebook_id = ?",
                            (source_file, notebook_id)
                        )
                    else:
                        cursor = conn.execute(
                            "DELETE FROM chunk_references WHERE source_file = ?",
                            (source_file,)
                        )
                    conn.commit()
                    deleted = cursor.rowcount
                    if deleted > 0:
                        logger.info(f"Deleted {deleted} chunk references for source {source_file}")
                    return deleted
            except Exception as e:
                logger.warning(f"Failed to delete chunk references: {e}")
                return 0
    
    def delete_by_notebook(self, notebook_id: int) -> int:
        """Delete all references for a notebook."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM chunk_references WHERE notebook_id = ?",
                        (notebook_id,)
                    )
                    conn.commit()
                    deleted = cursor.rowcount
                    if deleted > 0:
                        logger.info(f"Deleted {deleted} chunk references for notebook {notebook_id}")
                    return deleted
            except Exception as e:
                logger.warning(f"Failed to delete notebook chunk references: {e}")
                return 0
    
    def get_orphaned_chunks(self, all_chunk_ids: Set[str]) -> Set[str]:
        """Get chunks that have no notebook references."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Get all referenced chunks
                    cursor = conn.execute("SELECT DISTINCT chunk_id FROM chunk_references")
                    referenced_chunks = {row[0] for row in cursor.fetchall()}
                    # Return chunks that exist but aren't referenced
                    return all_chunk_ids - referenced_chunks
            except Exception as e:
                logger.warning(f"Failed to get orphaned chunks: {e}")
                return set()


# Global tracker instance
_reference_tracker: Optional[ChunkReferenceTracker] = None
_tracker_lock = threading.Lock()


def get_reference_tracker() -> ChunkReferenceTracker:
    """Get or create global chunk reference tracker instance."""
    global _reference_tracker
    if _reference_tracker is None:
        with _tracker_lock:
            if _reference_tracker is None:
                _reference_tracker = ChunkReferenceTracker()
    return _reference_tracker
