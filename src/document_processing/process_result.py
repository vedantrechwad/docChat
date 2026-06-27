"""Result type for document processing."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.document_processing.document_chunk import DocumentChunk


@dataclass
class ProcessResult:
    chunks: List[DocumentChunk]
    page_text: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
