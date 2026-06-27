"""Chunk cache - stores processed chunks by file checksum to avoid re-chunking."""

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.document_processing.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)


class ChunkCache:
    """Cache for document chunks keyed by file checksum."""
    
    def __init__(self, db_path: str = "./data/chunk_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for chunk cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_cache (
                    checksum TEXT PRIMARY KEY,
                    chunks_json TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    chunking_config TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checksum ON chunk_cache(checksum)")
            conn.commit()
    
    def get_chunks(self, checksum: str) -> Optional[List[DocumentChunk]]:
        """Retrieve cached chunks for a file checksum."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT chunks_json FROM chunk_cache WHERE checksum = ?",
                        (checksum,)
                    )
                    row = cursor.fetchone()
                    if row:
                        chunks_data = json.loads(row[0])
                        chunks = [DocumentChunk(**chunk) for chunk in chunks_data]
                        logger.info(f"Retrieved {len(chunks)} cached chunks for checksum {checksum[:8]}...")
                        return chunks
            except Exception as e:
                logger.warning(f"Failed to retrieve cached chunks: {e}")
        return None
    
    def store_chunks(
        self,
        checksum: str,
        chunks: List[DocumentChunk],
        source_file: str,
        chunking_config: Dict[str, Any]
    ) -> bool:
        """Store chunks in cache."""
        with self._lock:
            try:
                chunks_json = json.dumps([chunk.__dict__ for chunk in chunks])
                config_json = json.dumps(chunking_config)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO chunk_cache 
                           (checksum, chunks_json, source_file, chunking_config)
                           VALUES (?, ?, ?, ?)""",
                        (checksum, chunks_json, source_file, config_json)
                    )
                    conn.commit()
                logger.info(f"Cached {len(chunks)} chunks for checksum {checksum[:8]}...")
                return True
            except Exception as e:
                logger.warning(f"Failed to cache chunks: {e}")
                return False
    
    def invalidate(self, checksum: str) -> bool:
        """Remove chunks from cache."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM chunk_cache WHERE checksum = ?",
                        (checksum,)
                    )
                    conn.commit()
                    logger.info(f"Invalidated cache for checksum {checksum[:8]}...")
                    return cursor.rowcount > 0
            except Exception as e:
                logger.warning(f"Failed to invalidate cache: {e}")
                return False
    
    def clear_old_entries(self, days: int = 30) -> int:
        """Clear cache entries older than specified days."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """DELETE FROM chunk_cache 
                           WHERE datetime(created_at) < datetime('now', '-' || ? || ' days')""",
                        (days,)
                    )
                    conn.commit()
                    deleted = cursor.rowcount
                    if deleted > 0:
                        logger.info(f"Cleared {deleted} old cache entries")
                    return deleted
            except Exception as e:
                logger.warning(f"Failed to clear old cache entries: {e}")
                return 0


# Global cache instance
_chunk_cache: Optional[ChunkCache] = None
_cache_lock = threading.Lock()


def get_chunk_cache() -> ChunkCache:
    """Get or create global chunk cache instance."""
    global _chunk_cache
    if _chunk_cache is None:
        with _cache_lock:
            if _chunk_cache is None:
                _chunk_cache = ChunkCache()
    return _chunk_cache
