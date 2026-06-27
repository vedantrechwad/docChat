"""
Backend service helpers — chunking, ingest, indexing.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.document_processing.chunking_service import ChunkingService
from src.document_processing.document_chunk import DocumentChunk
from src.ingest.pipeline import (
    apply_chunking_to_processors,
    file_checksum,
    ingest_chunks,
    store_source_file,
)
from src.llm.model_registry import (
    CHUNK_PRESETS,
    EMBEDDING_MAX_TOKENS,
    get_active_model_name,
    recommend_chunk_preset,
)

logger = logging.getLogger(__name__)


def resolve_chunking(memory, llm_router, notebook_id: int, preset_override: Optional[str] = None) -> ChunkingService:
    settings = memory.get_chunking_settings(notebook_id)
    perf = memory.get_performance_mode()
    preset = preset_override or settings.get("preset", "auto")
    ingest_mode = "fast" if perf == "fast" else settings.get("ingest_mode", "quality")
    structure_first = ingest_mode == "fast"

    if preset == "auto":
        model = get_active_model_name(llm_router)
        if ingest_mode == "fast" or perf == "fast":
            preset = "compact"
        else:
            preset = recommend_chunk_preset(model)

    if preset == "custom":
        return ChunkingService.from_tokens(
            int(settings.get("chunk_tokens", 384)),
            int(settings.get("overlap_tokens", 100)),
            structure_first=structure_first,
        )
    return ChunkingService.from_preset(preset, structure_first=structure_first)


def get_chunking_profiles(llm_router) -> Dict[str, Any]:
    model = get_active_model_name(llm_router)
    recommended = recommend_chunk_preset(model)
    return {
        "presets": CHUNK_PRESETS,
        "embedding_max_tokens": EMBEDDING_MAX_TOKENS,
        "recommended": recommended,
        "active_model": model,
        "ingest_modes": ["fast", "quality"],
    }


def sort_chunks(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(r: Dict[str, Any]) -> Tuple[int, int]:
        c = r.get("citation", {})
        page = c.get("page_number")
        if page is None:
            page = 0
        return (page, c.get("chunk_index", 0))

    return sorted(results, key=sort_key)


def index_note_as_source(
    note: Dict[str, Any],
    notebook_id: int,
    embedding_generator,
    vector_db,
) -> None:
    """Index a note into Milvus for RAG retrieval."""
    source_name = f"note:{note['title']} (#{note['id']})"
    vector_db.delete_by_source(source_name, notebook_id=notebook_id)
    chunk = DocumentChunk(
        content=note["content"],
        source_file=source_name,
        source_type="note",
        chunk_index=0,
        metadata={"note_id": note["id"]},
    )
    embedded = embedding_generator.generate_embeddings([chunk])
    vector_db.insert_embeddings(embedded, notebook_id=notebook_id)


def reindex_source_content(
    source_id: int,
    content: str,
    page_text: Optional[Dict[str, str]],
    doc_processor,
    embedding_generator,
    vector_db,
    memory,
    source_info: Dict[str, Any],
) -> int:
    """Re-chunk and re-embed source after edit."""
    notebook_id = source_info["notebook_id"]
    source_name = source_info["name"]
    source_type = source_info.get("type", "Document").lower()

    vector_db.delete_by_source(source_name, notebook_id=notebook_id)

    if page_text:
        pages = [(int(k), v) for k, v in page_text.items()]
        pages.sort(key=lambda x: x[0])
        chunks = doc_processor.chunking.create_chunks_multi_page(
            pages=pages,
            source_file=source_name,
            source_type="pdf",
        )
        full_text = "\n\n".join(page_text.values())
    else:
        st = "md" if source_name.lower().endswith(".md") else "txt"
        if source_type == "clipboard":
            st = "clipboard"
        chunks = doc_processor.process_text_content(content, source_name, source_type=st)
        full_text = content

    embedded = embedding_generator.generate_embeddings(chunks)
    vector_db.insert_embeddings(embedded, notebook_id=notebook_id)

    memory.update_source(source_id, {
        "name": source_name,
        "type": source_info.get("type", "Document"),
        "size": f"{len(full_text)} chars",
        "chunks": len(chunks),
    }, notebook_id=notebook_id)

    sf = memory.get_source_file(source_id)
    if sf:
        memory.save_source_revision(source_id, sf.get("revision", 1), full_text)

    return len(chunks)
