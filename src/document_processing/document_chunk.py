import hashlib
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a processed document chunk with metadata for citations."""
    content: str
    source_file: str
    source_type: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self._generate_chunk_id()
        if self.metadata is None:
            self.metadata = {}

    def _generate_chunk_id(self) -> str:
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.source_type}_{self.chunk_index}_{content_hash}"

    def get_citation_info(self) -> Dict[str, Any]:
        citation = {
            'source': self.source_file,
            'type': self.source_type,
            'chunk_id': self.chunk_id,
            'chunk_index': self.chunk_index,
        }
        if self.page_number is not None:
            citation['page'] = self.page_number
        if self.start_char is not None or self.end_char is not None:
            citation['char_range'] = f"{self.start_char}-{self.end_char}"
        if self.metadata:
            citation.update(self.metadata)
        return citation
