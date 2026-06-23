"""
Notebook-scoped BM25 corpus index for full keyword recall beyond vector top-K.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.generation.hybrid_search import BM25Index

logger = logging.getLogger(__name__)

BM25_DIR = Path("./data/bm25")


class NotebookBM25Cache:
    """Lazy-built BM25 index per notebook from all Milvus chunks."""

    def __init__(self):
        self._indexes: Dict[int, BM25Index] = {}
        self._lock = threading.RLock()
        BM25_DIR.mkdir(parents=True, exist_ok=True)

    def _disk_path(self, notebook_id: int) -> Path:
        return BM25_DIR / f"{notebook_id}.json"

    def _load_disk(self, notebook_id: int) -> Optional[List[Dict[str, Any]]]:
        path = self._disk_path(notebook_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"BM25 disk load failed for notebook {notebook_id}: {e}")
            return None

    def _save_disk(self, notebook_id: int, docs: List[Dict[str, Any]]) -> None:
        path = self._disk_path(notebook_id)
        try:
            slim = [
                {"id": d.get("id"), "content": d.get("content", ""), "citation": d.get("citation", {})}
                for d in docs
            ]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(slim, f)
        except Exception as e:
            logger.warning(f"BM25 disk save failed: {e}")

    def invalidate(self, notebook_id: int) -> None:
        with self._lock:
            self._indexes.pop(notebook_id, None)
            path = self._disk_path(notebook_id)
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
            logger.info(f"BM25 cache invalidated for notebook {notebook_id}")

    def get_index(self, vector_db, notebook_id: int) -> BM25Index:
        with self._lock:
            if notebook_id in self._indexes and self._indexes[notebook_id].is_ready:
                return self._indexes[notebook_id]

        docs = self._load_disk(notebook_id)
        if docs is None:
            docs = vector_db.query_by_notebook(notebook_id=notebook_id, limit=10000)
            if docs:
                self._save_disk(notebook_id, docs)

        index = BM25Index()
        if docs:
            index.build_index(docs)

        with self._lock:
            self._indexes[notebook_id] = index
        logger.info(f"BM25 corpus built for notebook {notebook_id}: {len(docs or [])} docs")
        return index

    def search(
        self, vector_db, notebook_id: int, query: str, k: int = 20
    ) -> List[Tuple[int, float]]:
        index = self.get_index(vector_db, notebook_id)
        if not index.is_ready:
            return []
        return index.search(query, k=k)

    def get_document(self, vector_db, notebook_id: int, idx: int) -> Optional[Dict[str, Any]]:
        index = self.get_index(vector_db, notebook_id)
        return index.get_document(idx)


notebook_bm25_cache = NotebookBM25Cache()
