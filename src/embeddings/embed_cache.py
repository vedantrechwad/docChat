"""Content-hash embedding cache backed by SQLite."""

import hashlib
import logging
import sqlite3
import struct
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbedCache:
    """Cache embeddings by SHA-256 of chunk text + model name."""

    def __init__(self, db_path: str = "./data/embed_cache.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS embed_cache (
                content_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (content_hash, model_name)
            )"""
        )
        self.conn.commit()

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(
        self, texts: List[str], model_name: str
    ) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """Return cached vectors (None if miss) and indices of misses."""
        if not texts:
            return [], []
        hashes = [self.content_hash(t) for t in texts]
        placeholders = ",".join("?" * len(hashes))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT content_hash, dim, vector FROM embed_cache "
                f"WHERE model_name = ? AND content_hash IN ({placeholders})",
                [model_name, *hashes],
            ).fetchall()
        row_map = {r[0]: (r[1], r[2]) for r in rows}
        vectors: List[Optional[np.ndarray]] = []
        miss_indices: List[int] = []
        for i, h in enumerate(hashes):
            if h in row_map:
                dim, blob = row_map[h]
                vectors.append(np.array(struct.unpack(f"{dim}f", blob), dtype=np.float32))
            else:
                vectors.append(None)
                miss_indices.append(i)
        return vectors, miss_indices

    def put_many(self, texts: List[str], model_name: str, vectors: List[np.ndarray]) -> None:
        if not texts:
            return
        with self._lock:
            for text, vec in zip(texts, vectors):
                h = self.content_hash(text)
                dim = len(vec)
                blob = struct.pack(f"{dim}f", *vec.astype(np.float32).tolist())
                self.conn.execute(
                    "INSERT OR REPLACE INTO embed_cache (content_hash, model_name, dim, vector) VALUES (?, ?, ?, ?)",
                    (h, model_name, dim, blob),
                )
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()
