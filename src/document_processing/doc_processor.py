import logging

from typing import List, Dict, Any, Optional, Union

from pathlib import Path

from datetime import datetime



import pymupdf  # type: ignore



from src.document_processing.document_chunk import DocumentChunk

from src.document_processing.chunking_service import ChunkingService

from src.document_processing.process_result import ProcessResult



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)





class DocumentProcessor:

    def __init__(self, chunking: Optional[ChunkingService] = None):

        self.chunking = chunking or ChunkingService.from_preset("balanced")

        self.supported_formats = {'.pdf', '.txt', '.md'}



    def set_chunking(self, chunking: ChunkingService) -> None:

        self.chunking = chunking



    def process_document(self, file_path: str) -> ProcessResult:

        path_obj = Path(file_path)



        if not path_obj.exists():

            raise FileNotFoundError(f"File not found: {path_obj}")

        if path_obj.suffix.lower() not in self.supported_formats:

            raise ValueError(f"Unsupported file format: {path_obj.suffix}")



        logger.info(f"Processing document: {path_obj.name}")



        try:

            if path_obj.suffix.lower() == '.pdf':

                return self._process_pdf(path_obj)

            elif path_obj.suffix.lower() in {'.txt', '.md'}:

                chunks = self._process_text_file(path_obj)

                return ProcessResult(chunks=chunks)

            return ProcessResult(chunks=[])



        except Exception as e:

            logger.error(f"Error processing {path_obj.name}: {str(e)}")

            raise



    def process_text_content(

        self,

        text: str,

        source_file: str,

        source_type: str = "txt",

        additional_metadata: Optional[Dict[str, Any]] = None,

    ) -> List[DocumentChunk]:

        return self.chunking.create_chunks(

            text=text,

            source_file=source_file,

            source_type=source_type,

            additional_metadata=additional_metadata,

        )



    def extract_pdf_pages(self, file_path: Path) -> List[tuple[int, str]]:

        """Extract text per page from PDF (legacy helper)."""

        pages: List[tuple[int, str]] = []

        doc = pymupdf.open(file_path)

        try:

            for page_num in range(len(doc)):

                page = doc.load_page(page_num)

                text = page.get_text()

                if text.strip():

                    pages.append((page_num + 1, text))

        finally:

            doc.close()

        return pages



    def _extract_pdf_pages_raw(self, doc) -> tuple[List[tuple[int, str]], Dict[str, str], int]:

        pages: List[tuple[int, str]] = []

        page_text: Dict[str, str] = {}

        total_pages = len(doc)

        for page_num in range(total_pages):

            page = doc.load_page(page_num)

            text = page.get_text()

            if text.strip():

                p = page_num + 1

                pages.append((p, text))

                page_text[str(p)] = text

        return pages, page_text, total_pages



    def _process_pdf(self, file_path: Path) -> ProcessResult:

        try:

            doc = pymupdf.open(file_path)

            pages, page_text, total_pages = self._extract_pdf_pages_raw(doc)

            doc.close()



            metadata = {

                'total_pages': total_pages,

                'processed_at': datetime.now().isoformat(),

            }

            if self.chunking.structure_first:

                chunks = self.chunking.create_chunks_structured_pages(

                    pages=pages,

                    source_file=file_path.name,

                    source_type='pdf',

                    additional_metadata=metadata,

                )

            else:

                chunks = self.chunking.create_chunks_multi_page(

                    pages=pages,

                    source_file=file_path.name,

                    source_type='pdf',

                    additional_metadata=metadata,

                )

            logger.info(f"Processed PDF: {len(chunks)} chunks from {total_pages} pages")

            return ProcessResult(chunks=chunks, page_text=page_text, metadata=metadata)



        except Exception as e:

            logger.error(f"Error processing PDF {file_path}: {str(e)}")

            raise



    def _process_text_file(self, file_path: Path) -> List[DocumentChunk]:

        try:

            with open(file_path, 'r', encoding='utf-8') as file:

                content = file.read()



            metadata = {

                'file_size': file_path.stat().st_size,

                'encoding': 'utf-8',

                'processed_at': datetime.now().isoformat(),

            }



            source_type = 'md' if file_path.suffix.lower() == '.md' else 'txt'

            if self.chunking.structure_first:

                return self.chunking.create_chunks_structured_text(

                    content, file_path.name, source_type=source_type, additional_metadata=metadata,

                )

            chunks = self.chunking.create_chunks(

                content,

                file_path.name,

                source_type=source_type,

                additional_metadata=metadata,

            )



            logger.info(f"Processed text file: {len(chunks)} chunks")

            return chunks



        except Exception as e:

            logger.error(f"Error processing text file {file_path}: {str(e)}")

            raise



    def batch_process(self, file_paths: List[str]) -> List[DocumentChunk]:

        all_chunks = []

        for file_path in file_paths:

            try:

                result = self.process_document(file_path)

                all_chunks.extend(result.chunks)

                logger.info(f"Successfully processed {file_path}: {len(result.chunks)} chunks")

            except Exception as e:

                logger.error(f"Failed to process {file_path}: {str(e)}")

                continue



        logger.info(f"Batch processing complete: {len(all_chunks)} total chunks from {len(file_paths)} files")

        return all_chunks


