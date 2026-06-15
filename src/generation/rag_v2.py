"""
RAG Generator — Retrieval-Augmented Generation with citations.

Retrieves relevant document chunks via vector search, then generates
a cited response using the LLM router.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.llm.llm_router import LLMRouter
from src.vector_database.milvus_vector_db import MilvusVectorDB
from src.embeddings.embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """RAG Generator using the LLM Router for provider-agnostic generation."""

    SYSTEM_PROMPT = """You are an AI assistant that answers questions based on provided source material. Follow these rules:
1. For each factual claim, include the citation reference number in square brackets [1], [2], etc.
2. Only use information from the provided context — do not add external knowledge.
3. If you cannot find relevant information in the context, say so clearly.
4. Be precise and accurate in your citations.
5. When multiple sources support the same point, list all relevant citations like [1], [2]."""

    def __init__(
        self,
        llm_router: LLMRouter,
        embedding_generator: EmbeddingGenerator,
        vector_db: MilvusVectorDB,
    ):
        self.llm_router = llm_router
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        logger.info("RAG Generator initialized")

    def generate_response(
        self,
        query: str,
        max_chunks: int = 8,
        max_context_chars: int = 4000,
        top_k: int = 10,
        conversation_context: str = "",
    ) -> RAGResult:
        """Generate a cited response using RAG."""

        if not query.strip():
            return RAGResult(query=query, response="Please provide a question.", sources_used=[], retrieval_count=0)

        try:
            # 1. Retrieve relevant chunks
            query_vector = self.embedding_generator.generate_query_embedding(query)
            search_results = self.vector_db.search(query_vector=query_vector.tolist(), limit=top_k)

            if not search_results:
                return RAGResult(
                    query=query,
                    response="I couldn't find any relevant information in the available documents to answer your question.",
                    sources_used=[],
                    retrieval_count=0,
                )

            # 2. Format context with citations
            context, sources_info = self._format_context(search_results, max_chunks, max_context_chars)

            # 3. Build prompt
            prompt = self._build_prompt(query, context, conversation_context)

            # 4. Generate response
            llm_response = self.llm_router.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=2000,
            )

            result = RAGResult(
                query=query,
                response=llm_response.content,
                sources_used=sources_info,
                retrieval_count=len(search_results),
            )

            logger.info(f"Response generated with {len(sources_info)} sources via {llm_response.provider}/{llm_response.model}")
            return result

        except Exception as e:
            logger.error(f"RAG error: {e}")
            return RAGResult(
                query=query,
                response=f"I encountered an error while processing your question: {str(e)}",
                sources_used=[],
                retrieval_count=0,
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
            citation = result["citation"]
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
                "chunk_id": result["id"],
                "relevance_score": result["score"],
            })

        return "\n\n".join(context_parts), sources_info

    def _build_prompt(self, query: str, context: str, conversation_context: str = "") -> str:
        """Build the RAG prompt."""
        conv_section = ""
        if conversation_context:
            conv_section = f"\nPREVIOUS CONVERSATION:\n{conversation_context}\n"

        return f"""{conv_section}
CONTEXT (with citation references):
{context}

QUESTION: {query}

Provide a comprehensive answer with proper citations. Every factual statement must be supported by a citation reference."""
