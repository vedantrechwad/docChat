"""
Hybrid Search — BM25 keyword search combined with vector similarity.

Implements Reciprocal Rank Fusion (RRF) to merge results from both
retrieval methods, significantly improving recall for specific terms.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Index:
    """
    Lightweight BM25 keyword index using bm25s.
    Rebuilt on-demand from Milvus query results.
    """

    def __init__(self):
        self._corpus: List[Dict[str, Any]] = []
        self._retriever = None
        self._corpus_texts: List[str] = []

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """Build BM25 index from a list of documents with 'content' and 'id' fields."""
        if not documents:
            self._retriever = None
            return

        try:
            import bm25s

            self._corpus = documents
            self._corpus_texts = [doc.get("content", "") for doc in documents]

            self._retriever = bm25s.BM25(corpus=self._corpus_texts)
            self._retriever.index(bm25s.tokenize(self._corpus_texts))

            logger.info(f"BM25 index built with {len(documents)} documents")
        except ImportError:
            logger.warning("bm25s not installed — BM25 search disabled")
            self._retriever = None
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self._retriever = None

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search and return (doc_index, score) pairs."""
        if not self._retriever or not self._corpus:
            return []

        try:
            import bm25s

            tokenized_query = bm25s.tokenize(query)
            results, scores = self._retriever.retrieve(tokenized_query, k=min(k, len(self._corpus)))

            # results and scores are 2D arrays (batch), we only have 1 query
            pairs = []
            if len(results) > 0 and len(scores) > 0:
                for idx, score in zip(results[0], scores[0]):
                    if score > 0:
                        pairs.append((int(idx), float(score)))

            return pairs
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []

    def get_document(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a document by its index."""
        if 0 <= idx < len(self._corpus):
            return self._corpus[idx]
        return None

    @property
    def is_ready(self) -> bool:
        return self._retriever is not None and len(self._corpus) > 0


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Tuple[int, float]],
    bm25_docs: "BM25Index",
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Merge vector search and BM25 results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each retrieval method.
    Higher k → more emphasis on lower-ranked results.

    Args:
        vector_results: Results from vector similarity search
        bm25_results: (doc_index, score) pairs from BM25
        bm25_docs: BM25Index instance to look up documents
        k: RRF parameter (default 60, standard value)

    Returns:
        Merged and re-scored results
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Score vector results
    for rank, doc in enumerate(vector_results):
        doc_id = doc.get("id", f"vec_{rank}")
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        doc_map[doc_id] = doc

    # Score BM25 results
    for rank, (doc_idx, bm25_score) in enumerate(bm25_results):
        bm25_doc = bm25_docs.get_document(doc_idx)
        if not bm25_doc:
            continue

        doc_id = bm25_doc.get("id", f"bm25_{rank}")
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        if doc_id not in doc_map:
            doc_map[doc_id] = bm25_doc

    # Sort by fused score (higher is better)
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    merged = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        doc["rrf_score"] = scores[doc_id]
        merged.append(doc)

    return merged
