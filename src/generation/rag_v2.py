"""
RAG Generator v2 - Uses LLM Router for model-agnostic generation.

Replaces the OpenAI-only RAGGenerator with one that works with
Ollama, Groq, or any provider through the LLM Router abstraction.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.llm.llm_router import LLMRouter, TaskType
from src.vector_database.milvus_vector_db import MilvusVectorDB
from src.embeddings.embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Represents the result of RAG generation with citations."""
    query: str
    response: str
    sources_used: List[Dict[str, Any]]
    retrieval_count: int
    generation_tokens: Optional[int] = None

    def get_citation_summary(self) -> str:
        if not self.sources_used:
            return "No sources cited"

        source_summary = []
        for source in self.sources_used:
            source_info = f"• {source.get('source_file', 'Unknown')} ({source.get('source_type', 'unknown')})"
            if source.get('page_number'):
                source_info += f" - Page {source['page_number']}"
            source_summary.append(source_info)

        return "\n".join(source_summary)


class RAGGeneratorV2:
    """
    RAG Generator using the LLM Router for provider-agnostic generation.
    
    Works with Ollama (local), Groq (cloud), or OpenAI through the same interface.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        embedding_generator: EmbeddingGenerator,
        vector_db: MilvusVectorDB,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        self.llm_router = llm_router
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info("RAG Generator V2 initialized with LLM Router")

    def generate_response(
        self,
        query: str,
        max_chunks: int = 8,
        max_context_chars: int = 4000,
        top_k: int = 10,
        conversation_context: str = "",
    ) -> RAGResult:
        """Generate a response with citations using RAG."""

        if not query.strip():
            return RAGResult(
                query=query,
                response="Please provide a valid question.",
                sources_used=[],
                retrieval_count=0,
            )

        try:
            logger.info(f"Generating response for: '{query[:50]}...'")

            # Step 1: Retrieve relevant chunks
            query_vector = self.embedding_generator.generate_query_embedding(query)
            search_results = self.vector_db.search(
                query_vector=query_vector.tolist(),
                limit=top_k,
            )

            if not search_results:
                return RAGResult(
                    query=query,
                    response="I couldn't find any relevant information in the available documents to answer your question.",
                    sources_used=[],
                    retrieval_count=0,
                )

            # Step 2: Format context with citations
            context, sources_info = self._format_context_with_citations(
                search_results, max_chunks, max_context_chars
            )

            # Step 3: Create citation-aware prompt
            prompt = self._create_rag_prompt(query, context, conversation_context)

            # Step 4: Generate response via LLM Router
            llm_response = self.llm_router.generate(
                prompt=prompt,
                task_type=TaskType.CHAT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Step 5: Create result
            rag_result = RAGResult(
                query=query,
                response=llm_response.content,
                sources_used=sources_info,
                retrieval_count=len(search_results),
                generation_tokens=llm_response.usage.get("completion_tokens"),
            )

            logger.info(f"Response generated with {len(sources_info)} sources via {llm_response.provider}/{llm_response.model}")
            return rag_result

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return RAGResult(
                query=query,
                response=f"I encountered an error while processing your question: {str(e)}",
                sources_used=[],
                retrieval_count=0,
            )

    def _format_context_with_citations(
        self,
        search_results: List[Dict[str, Any]],
        max_chunks: int,
        max_context_chars: int,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Format retrieved chunks with citation references."""

        context_parts = []
        sources_info = []
        total_chars = 0

        for i, result in enumerate(search_results[:max_chunks]):
            citation_info = result['citation']
            source_file = citation_info.get('source_file', 'Unknown Source')
            source_type = citation_info.get('source_type', 'unknown')
            page_number = citation_info.get('page_number')

            citation_ref = f"[{i + 1}]"
            chunk_content = result['content']
            chunk_text = f"{citation_ref} {chunk_content}"

            if total_chars + len(chunk_text) > max_context_chars and context_parts:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

            source_info = {
                'reference': citation_ref,
                'source_file': source_file,
                'source_type': source_type,
                'page_number': page_number,
                'chunk_id': result['id'],
                'relevance_score': result['score'],
                'content': chunk_content,
            }
            sources_info.append(source_info)

        return '\n\n'.join(context_parts), sources_info

    def _create_rag_prompt(self, query: str, context: str, conversation_context: str = "") -> str:
        """Create a citation-aware RAG prompt."""

        conv_section = ""
        if conversation_context:
            conv_section = f"""
PREVIOUS CONVERSATION (for context):
{conversation_context}

"""

        prompt = f"""You are an AI study assistant that answers questions based on provided source material. You must follow these citation rules:

CITATION REQUIREMENTS:
1. For each factual claim in your answer, include the citation reference number in square brackets [1], [2], etc.
2. Only use information from the provided context - do not add external knowledge
3. If you cannot find relevant information in the context, say so clearly
4. Be precise and accurate in your citations
5. When multiple sources support the same point, list all relevant citations like this [1], [2], [3].
{conv_section}
CONTEXT (with citation references):
{context}

QUESTION: {query}

Please provide a comprehensive answer with proper citations. Make sure every factual statement is supported by a citation reference."""

        return prompt

    def generate_summary(
        self,
        max_chunks: int = 15,
        summary_length: str = "medium",
    ) -> RAGResult:
        """Generate a summary of all stored documents."""

        try:
            summary_query = "main topics key findings important information overview"
            query_vector = self.embedding_generator.generate_query_embedding(summary_query)
            search_results = self.vector_db.search(
                query_vector=query_vector.tolist(),
                limit=max_chunks,
            )

            if not search_results:
                return RAGResult(
                    query="Document Summary",
                    response="No documents available for summarization.",
                    sources_used=[],
                    retrieval_count=0,
                )

            context, sources_info = self._format_context_with_citations(
                search_results, max_chunks, 6000
            )

            length_instructions = {
                'short': "Provide a concise 2-3 paragraph summary highlighting the most important points.",
                'medium': "Provide a comprehensive 4-5 paragraph summary covering key topics and findings.",
                'long': "Provide a detailed summary with multiple sections covering all major topics and supporting details.",
            }

            summary_prompt = f"""You are tasked with creating a summary of the provided document content. Follow these guidelines:

1. {length_instructions.get(summary_length, length_instructions['medium'])}
2. Include citations [1], [2], etc. for all factual claims
3. Organize information logically with clear topics
4. Focus on the most important and relevant information
5. Maintain accuracy and cite sources properly

DOCUMENT CONTENT (with citation references):
{context}

Please provide a well-structured summary with proper citations:"""

            llm_response = self.llm_router.generate(
                prompt=summary_prompt,
                task_type=TaskType.SUMMARIZE,
                temperature=0.2,
                max_tokens=3000,
            )

            return RAGResult(
                query="Document Summary",
                response=llm_response.content,
                sources_used=sources_info,
                retrieval_count=len(search_results),
            )

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return RAGResult(
                query="Document Summary",
                response=f"Error generating summary: {str(e)}",
                sources_used=[],
                retrieval_count=0,
            )
