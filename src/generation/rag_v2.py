"""
RAG Generator V2 — Advanced Retrieval-Augmented Generation.

Features:
- Adaptive context window based on active LLM model
- Notebook-isolated vector search
- Hybrid search (BM25 keyword + vector similarity + RRF fusion)
- HyDE query rewriting for better retrieval
- LLM-based reranking of retrieved chunks
- Embedding caching for repeated queries
- Optimized prompts for both small local models and large API models
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.llm.llm_router import LLMRouter
from src.vector_database.milvus_vector_db import MilvusVectorDB
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.generation.hybrid_search import BM25Index, reciprocal_rank_fusion
from src.generation.notebook_bm25 import notebook_bm25_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Context Profiles ──────────────────────────────────────────────────────────

CONTEXT_PROFILES = {
    (0, 4096): {
        "max_context_chars": 2000,
        "max_chunks": 3,
        "max_tokens": 1000,
        "top_k": 8,
        "use_hyde": False,         # HyDE is too expensive for small models
        "use_reranking": False,    # Reranking uses LLM tokens
        "label": "small",
    },
    (4097, 8192): {
        "max_context_chars": 4000,
        "max_chunks": 5,
        "max_tokens": 1500,
        "top_k": 12,
        "use_hyde": False,
        "use_reranking": True,
        "label": "medium",
    },
    (8193, 32768): {
        "max_context_chars": 8000,
        "max_chunks": 7,
        "max_tokens": 2000,
        "top_k": 15,
        "use_hyde": True,
        "use_reranking": True,
        "label": "large",
    },
    (32769, float("inf")): {
        "max_context_chars": 12000,
        "max_chunks": 10,
        "max_tokens": 3000,
        "top_k": 20,
        "use_hyde": True,
        "use_reranking": True,
        "label": "api",
    },
}


def get_context_profile(context_size: int) -> Dict[str, Any]:
    """Get the appropriate context profile for a given model context size."""
    for (min_ctx, max_ctx), profile in CONTEXT_PROFILES.items():
        if min_ctx <= context_size <= max_ctx:
            return profile
    return list(CONTEXT_PROFILES.values())[0]


@dataclass
class RAGResult:
    """Result of RAG generation with citations."""
    query: str
    response: str
    sources_used: List[Dict[str, Any]]
    retrieval_count: int

    def get_citation_summary(self) -> str:
        if not self.sources_used:
            return "No sources cited"
        lines = []
        for s in self.sources_used:
            info = f"• {s.get('source_file', 'Unknown')} ({s.get('source_type', 'unknown')})"
            if s.get('page_number'):
                info += f" — Page {s['page_number']}"
            lines.append(info)
        return "\n".join(lines)


class RAGGeneratorV2:
    """Advanced RAG Generator with hybrid search, HyDE, and reranking."""

    SYSTEM_PROMPT = """You are an AI assistant that answers questions using provided source material.

Rules:
1. Cite sources with [1], [2], etc. for each factual claim.
2. Only use information from the provided context.
3. If no relevant info found, say so clearly.
4. When multiple sources support a point, list all: [1], [2]."""

    def __init__(
        self,
        llm_router: LLMRouter,
        embedding_generator: EmbeddingGenerator,
        vector_db: MilvusVectorDB,
    ):
        self.llm_router = llm_router
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.bm25_index = BM25Index()
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 100
        logger.info("RAG Generator V2 initialized (hybrid search + adaptive context)")

    def _get_adaptive_settings(self) -> Dict[str, Any]:
        """Get context/chunking settings adaptive to the active model."""
        ctx_size = self.llm_router.get_model_context_size()
        profile = get_context_profile(ctx_size)
        logger.info(f"Adaptive profile: {profile['label']} (model ctx: {ctx_size})")
        return profile

    def _get_cached_embedding(self, query: str) -> np.ndarray:
        """Get query embedding with LRU caching."""
        if query in self._embedding_cache:
            return self._embedding_cache[query]

        embedding = self.embedding_generator.generate_query_embedding(query)

        # Evict oldest if cache is full
        if len(self._embedding_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]

        self._embedding_cache[query] = embedding
        return embedding

    def _hyde_rewrite(self, query: str) -> str:
        """
        HyDE: Hypothetical Document Embeddings.
        Generate a hypothetical answer to use as the search query,
        which produces better embeddings than the raw question.
        """
        try:
            response = self.llm_router.generate(
                prompt=f"Write a short factual paragraph that would answer this question: {query}",
                system_prompt="You are a helpful assistant. Write a brief, factual paragraph as if answering from a document. Be specific and use technical terms.",
                temperature=0.3,
                max_tokens=200,
            )
            hyde_text = response.content.strip()
            if hyde_text and len(hyde_text) > 20:
                logger.info(f"HyDE rewrite: '{query[:50]}...' -> '{hyde_text[:80]}...'")
                return hyde_text
        except Exception as e:
            logger.warning(f"HyDE rewrite failed: {e}")

        return query  # Fallback to original query

    def _rerank_chunks(
        self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Use the LLM to rerank retrieved chunks by relevance.
        Scores each chunk on a 1-10 scale and re-orders.
        """
        if len(chunks) <= top_n:
            return chunks

        try:
            # Build a compact scoring prompt
            chunk_descriptions = []
            for i, chunk in enumerate(chunks[:15]):  # Limit to avoid token overflow
                preview = chunk["content"][:200].replace("\n", " ")
                chunk_descriptions.append(f"[{i}] {preview}")

            chunks_text = "\n".join(chunk_descriptions)

            response = self.llm_router.generate(
                prompt=f"Question: {query}\n\nRank these text passages by relevance (most relevant first). Return ONLY a comma-separated list of passage numbers, e.g.: 3,1,5,0,2\n\n{chunks_text}",
                system_prompt="You are a relevance ranker. Return ONLY comma-separated passage numbers ordered by relevance. No explanation.",
                temperature=0.0,
                max_tokens=100,
            )

            # Parse the ranking
            ranking_text = response.content.strip()
            # Extract numbers from the response
            import re
            numbers = re.findall(r'\d+', ranking_text)
            ranked_indices = []
            for n in numbers:
                idx = int(n)
                if 0 <= idx < len(chunks) and idx not in ranked_indices:
                    ranked_indices.append(idx)

            if ranked_indices:
                reranked = [chunks[i] for i in ranked_indices[:top_n]]
                # Add any chunks that weren't ranked (in case LLM missed some)
                for i, chunk in enumerate(chunks):
                    if i not in ranked_indices and len(reranked) < top_n:
                        reranked.append(chunk)
                logger.info(f"Reranked {len(chunks)} chunks -> top {len(reranked)}")
                return reranked

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")

        return chunks[:top_n]

    def _build_bm25_from_vector_results(
        self, vector_results: List[Dict[str, Any]]
    ) -> None:
        """Build BM25 index from vector search results for hybrid fusion."""
        docs = []
        for r in vector_results:
            docs.append({
                "id": r["id"],
                "content": r["content"],
                "score": r["score"],
                "citation": r["citation"],
                "metadata": r.get("metadata", {}),
                "embedding_model": r.get("embedding_model", ""),
            })
        self.bm25_index.build_index(docs)

    def generate_response(
        self,
        query: str,
        notebook_id: Optional[int] = None,
        conversation_context: str = "",
    ) -> RAGResult:
        """Generate a cited response using advanced RAG pipeline."""

        if not query.strip():
            return RAGResult(
                query=query, response="Please provide a question.",
                sources_used=[], retrieval_count=0,
            )

        try:
            settings = self._get_adaptive_settings()
            max_chunks = settings["max_chunks"]
            max_context_chars = settings["max_context_chars"]
            max_tokens = settings["max_tokens"]
            top_k = settings["top_k"]
            use_hyde = settings["use_hyde"]
            use_reranking = settings["use_reranking"]

            # ── Step 1: Query processing ───────────────────────────────
            search_query = query
            if use_hyde:
                search_query = self._hyde_rewrite(query)

            # ── Step 2: Vector search (notebook-isolated) ──────────────
            # Cache embedding under original query (HyDE text is non-deterministic)
            if use_hyde and search_query != query:
                # Generate fresh embedding for HyDE text (don't cache it)
                query_vector = self.embedding_generator.generate_query_embedding(search_query)
            else:
                query_vector = self._get_cached_embedding(query)
            vector_results = self.vector_db.search(
                query_vector=query_vector.tolist(),
                limit=top_k,
                notebook_id=notebook_id,
            )

            if not vector_results:
                return RAGResult(
                    query=query,
                    response="I couldn't find any relevant information in the available documents to answer your question.",
                    sources_used=[], retrieval_count=0,
                )

            # ── Step 3: Hybrid search (full-corpus BM25 + RRF) ─────────
            try:
                if notebook_id is not None:
                    bm25_results = notebook_bm25_cache.search(
                        self.vector_db, notebook_id, query, k=top_k * 2
                    )
                    if bm25_results:
                        corpus_index = notebook_bm25_cache.get_index(self.vector_db, notebook_id)
                        merged = reciprocal_rank_fusion(
                            vector_results, bm25_results, corpus_index
                        )
                        vector_results = merged
                        logger.info(f"Hybrid search (corpus BM25): {len(merged)} fused results")
                else:
                    self._build_bm25_from_vector_results(vector_results)
                    if self.bm25_index.is_ready:
                        bm25_results = self.bm25_index.search(query, k=top_k)
                        if bm25_results:
                            merged = reciprocal_rank_fusion(
                                vector_results, bm25_results, self.bm25_index
                            )
                            vector_results = merged
                            logger.info(f"Hybrid search: {len(merged)} fused results")
            except Exception as e:
                logger.warning(f"BM25 hybrid search failed (falling back to vector only): {e}")

            # ── Step 4: Reranking ──────────────────────────────────────
            if use_reranking and len(vector_results) > max_chunks:
                vector_results = self._rerank_chunks(query, vector_results, max_chunks)

            # ── Step 5: Format context with citations ──────────────────
            context, sources_info = self._format_context(
                vector_results, max_chunks, max_context_chars
            )

            # ── Step 6: Build prompt and generate ──────────────────────
            prompt = self._build_prompt(query, context, conversation_context)

            llm_response = self.llm_router.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=max_tokens,
            )

            result = RAGResult(
                query=query,
                response=llm_response.content,
                sources_used=sources_info,
                retrieval_count=len(vector_results),
            )

            logger.info(
                f"Response generated with {len(sources_info)} sources "
                f"via {llm_response.provider}/{llm_response.model} "
                f"(profile: {settings['label']}, hyde: {use_hyde}, rerank: {use_reranking})"
            )
            return result

        except Exception as e:
            logger.error(f"RAG error: {e}")
            return RAGResult(
                query=query,
                response=f"I encountered an error while processing your question: {str(e)}",
                sources_used=[], retrieval_count=0,
            )

    def _format_context(
        self, search_results: List[Dict[str, Any]],
        max_chunks: int, max_context_chars: int,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Format retrieved chunks with citation references."""
        context_parts = []
        sources_info = []
        total_chars = 0

        for i, result in enumerate(search_results[:max_chunks]):
            citation = result.get("citation", {})
            ref = f"[{i + 1}]"
            chunk_text = f"{ref} {result['content']}"

            if total_chars + len(chunk_text) > max_context_chars and context_parts:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

            sources_info.append({
                "reference": ref,
                "source_file": citation.get("source_file", "Unknown"),
                "source_type": citation.get("source_type", "unknown"),
                "page_number": citation.get("page_number"),
                "chunk_id": result.get("id", ""),
                "chunk_index": citation.get("chunk_index"),
                "relevance_score": result.get("score", result.get("rrf_score", 0)),
                "text": result["content"][:300],
            })

        return "\n\n".join(context_parts), sources_info

    def _build_prompt(self, query: str, context: str, conversation_context: str = "") -> str:
        """Build the RAG prompt — kept concise for small models."""
        conv_section = ""
        if conversation_context:
            conv_section = f"\nPREVIOUS CONVERSATION:\n{conversation_context}\n"

        return f"""{conv_section}
CONTEXT (with citation references):
{context}

QUESTION: {query}

Provide a comprehensive answer with proper citations. Every factual statement must be supported by a citation reference."""

    def prepare_response(
        self,
        query: str,
        notebook_id: Optional[int] = None,
        conversation_context: str = "",
    ) -> Tuple[str, str, List[Dict[str, Any]], int, Dict[str, Any]]:
        """Run retrieval and return prompt, system prompt, sources, count, settings."""
        if not query.strip():
            return "", self.SYSTEM_PROMPT, [], 0, {}

        settings = self._get_adaptive_settings()
        max_chunks = settings["max_chunks"]
        max_context_chars = settings["max_context_chars"]
        max_tokens = settings["max_tokens"]
        top_k = settings["top_k"]
        use_hyde = settings["use_hyde"]
        use_reranking = settings["use_reranking"]

        search_query = self._hyde_rewrite(query) if use_hyde else query

        if use_hyde and search_query != query:
            query_vector = self.embedding_generator.generate_query_embedding(search_query)
        else:
            query_vector = self._get_cached_embedding(query)

        vector_results = self.vector_db.search(
            query_vector=query_vector.tolist(),
            limit=top_k,
            notebook_id=notebook_id,
        )

        if not vector_results:
            return "", self.SYSTEM_PROMPT, [], 0, settings

        try:
            if notebook_id is not None:
                bm25_results = notebook_bm25_cache.search(
                    self.vector_db, notebook_id, query, k=top_k * 2
                )
                if bm25_results:
                    corpus_index = notebook_bm25_cache.get_index(self.vector_db, notebook_id)
                    vector_results = reciprocal_rank_fusion(
                        vector_results, bm25_results, corpus_index
                    )
        except Exception as e:
            logger.warning(f"BM25 hybrid search failed: {e}")

        if use_reranking and len(vector_results) > max_chunks:
            vector_results = self._rerank_chunks(query, vector_results, max_chunks)

        context, sources_info = self._format_context(
            vector_results, max_chunks, max_context_chars
        )
        prompt = self._build_prompt(query, context, conversation_context)
        settings["max_tokens"] = max_tokens
        return prompt, self.SYSTEM_PROMPT, sources_info, len(vector_results), settings

    def generate_response_stream(
        self,
        query: str,
        notebook_id: Optional[int] = None,
        conversation_context: str = "",
    ):
        """Yield (event_type, data) tuples for SSE streaming."""
        if not query.strip():
            yield ("error", {"message": "Please provide a question."})
            return

        try:
            prompt, system_prompt, sources_info, retrieval_count, settings = self.prepare_response(
                query, notebook_id, conversation_context
            )

            if not prompt:
                yield ("token", {"text": "I couldn't find any relevant information in the available documents to answer your question."})
                yield ("done", {"sources_used": [], "retrieval_count": 0})
                return

            yield ("meta", {
                "sources_used": sources_info,
                "retrieval_count": retrieval_count,
                "profile": settings.get("label", ""),
            })

            for token in self.llm_router.generate_stream(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=settings.get("max_tokens", 2000),
            ):
                yield ("token", {"text": token})

            yield ("done", {"sources_used": sources_info, "retrieval_count": retrieval_count})

        except Exception as e:
            logger.error(f"RAG stream error: {e}")
            yield ("error", {"message": str(e)})
