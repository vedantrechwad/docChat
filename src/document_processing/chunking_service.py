"""
Centralized token-aware chunking for all ingest paths.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.document_processing.document_chunk import DocumentChunk
from src.llm.model_registry import (
    CHUNK_PRESETS,
    EMBEDDING_MAX_TOKENS,
    get_chunk_preset,
    tokens_to_chars,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token for English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class ChunkingService:
    """Single chunking implementation used by PDF, web, YouTube, clipboard."""

    def __init__(
        self,
        preset: str = "balanced",
        chunk_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
        structure_first: bool = False,
    ):
        self.structure_first = structure_first
        if chunk_tokens is not None and overlap_tokens is not None:
            self.chunk_tokens = min(chunk_tokens, EMBEDDING_MAX_TOKENS)
            self.overlap_tokens = min(overlap_tokens, self.chunk_tokens // 2)
            self.preset_name = "custom"
        else:
            cfg = get_chunk_preset(preset)
            self.chunk_tokens = cfg["chunk_tokens"]
            self.overlap_tokens = cfg["overlap_tokens"]
            self.preset_name = preset if preset in CHUNK_PRESETS else "balanced"

        self.chunk_size = tokens_to_chars(self.chunk_tokens)
        self.chunk_overlap = tokens_to_chars(self.overlap_tokens)

    @classmethod
    def from_preset(cls, preset: str, structure_first: bool = False) -> "ChunkingService":
        return cls(preset=preset, structure_first=structure_first)

    @classmethod
    def from_tokens(cls, chunk_tokens: int, overlap_tokens: int, structure_first: bool = False) -> "ChunkingService":
        return cls(
            preset="custom",
            chunk_tokens=min(chunk_tokens, EMBEDDING_MAX_TOKENS),
            overlap_tokens=overlap_tokens,
            structure_first=structure_first,
        )

    def validate_chunk(self, text: str) -> bool:
        return estimate_tokens(text) <= EMBEDDING_MAX_TOKENS

    def create_chunks(
        self,
        text: str,
        source_file: str,
        source_type: str,
        page_number: Optional[int] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        start_index: int = 0,
    ) -> List[DocumentChunk]:
        if not text or not text.strip():
            return []

        chunks: List[DocumentChunk] = []
        start = 0
        chunk_index = start_index

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                boundary = -1
                for sep in ("\n\n", ". ", ".\n", "\n"):
                    pos = text.rfind(sep, start, end)
                    if pos > start + int(self.chunk_size * 0.4):
                        boundary = pos + len(sep)
                        break
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                if not self.validate_chunk(chunk_text):
                    # Shrink chunk if over token limit
                    max_chars = tokens_to_chars(EMBEDDING_MAX_TOKENS)
                    chunk_text = chunk_text[:max_chars].rsplit(" ", 1)[0] or chunk_text[:max_chars]
                    end = start + len(chunk_text)

                meta = dict(additional_metadata) if additional_metadata else {}
                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        source_file=source_file,
                        source_type=source_type,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=start + len(chunk_text) - 1,
                        metadata=meta,
                    )
                )
                chunk_index += 1

            start = max(end - self.chunk_overlap, start + 1)
            if start >= len(text):
                break

        return chunks

    def create_chunks_multi_page(
        self,
        pages: List[tuple[int, str]],
        source_file: str,
        source_type: str = "pdf",
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Chunk PDF pages with global chunk_index across pages."""
        all_chunks: List[DocumentChunk] = []
        global_index = 0
        for page_num, text in pages:
            if not text.strip():
                continue
            page_meta = dict(additional_metadata) if additional_metadata else {}
            page_meta["page_number"] = page_num
            page_chunks = self.create_chunks(
                text=text,
                source_file=source_file,
                source_type=source_type,
                page_number=page_num,
                additional_metadata=page_meta,
                start_index=global_index,
            )
            global_index += len(page_chunks)
            all_chunks.extend(page_chunks)
        return all_chunks

    @staticmethod
    def _is_section_heading(line: str) -> bool:
        if not line or len(line) > 120:
            return False
        stripped = line.strip()
        if re.match(r"^(chapter|section|part|appendix)\s+[\d\.]+", stripped, re.I):
            return True
        if re.match(r"^\d+(\.\d+)*[\.\)]?\s+\S", stripped):
            return True
        if stripped.isupper() and 4 < len(stripped) < 80 and stripped.count(" ") < 12:
            return True
        return False

    def _split_into_sections(self, text: str) -> List[str]:
        lines = text.split("\n")
        sections: List[str] = []
        current: List[str] = []
        for line in lines:
            if self._is_section_heading(line.strip()) and current:
                block = "\n".join(current).strip()
                if block:
                    sections.append(block)
                current = [line]
            else:
                current.append(line)
        if current:
            block = "\n".join(current).strip()
            if block:
                sections.append(block)
        return sections if len(sections) > 1 else [text]

    def create_chunks_structured_text(
        self,
        text: str,
        source_file: str,
        source_type: str,
        page_number: Optional[int] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        start_index: int = 0,
    ) -> List[DocumentChunk]:
        sections = self._split_into_sections(text)
        all_chunks: List[DocumentChunk] = []
        idx = start_index
        for section in sections:
            section_chunks = self.create_chunks(
                section, source_file, source_type, page_number, additional_metadata, start_index=idx,
            )
            idx += len(section_chunks)
            all_chunks.extend(section_chunks)
        return all_chunks

    def create_chunks_structured_pages(
        self,
        pages: List[tuple[int, str]],
        source_file: str,
        source_type: str = "pdf",
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        all_chunks: List[DocumentChunk] = []
        global_index = 0
        for page_num, text in pages:
            if not text.strip():
                continue
            page_meta = dict(additional_metadata) if additional_metadata else {}
            page_meta["page_number"] = page_num
            sections = self._split_into_sections(text)
            for section in sections:
                page_chunks = self.create_chunks(
                    text=section,
                    source_file=source_file,
                    source_type=source_type,
                    page_number=page_num,
                    additional_metadata=page_meta,
                    start_index=global_index,
                )
                global_index += len(page_chunks)
                all_chunks.extend(page_chunks)
        return all_chunks
