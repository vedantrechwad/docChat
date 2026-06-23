"""

Shared ingest pipeline — chunk, embed, store with deduplication.

"""



import hashlib

import logging

import shutil

from pathlib import Path

from typing import Any, Callable, Dict, List, Optional



from src.document_processing.chunking_service import ChunkingService

from src.ingest.ingest_jobs import IngestJobManager, JobStatus



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)



SOURCES_DIR = Path("./data/sources")





def file_checksum(path: Path) -> str:

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(lambda: f.read(65536), b""):

            h.update(block)

    return h.hexdigest()





def store_source_file(notebook_id: int, source_id: int, src_path: Path, original_name: str) -> Path:

    """Persist original file for later editing."""

    dest_dir = SOURCES_DIR / str(notebook_id) / str(source_id)

    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / original_name

    shutil.copy2(src_path, dest)

    return dest





def apply_chunking_to_processors(

    chunking: ChunkingService,

    doc_processor,

    web_scraper,

    youtube_extractor,

) -> None:

    doc_processor.set_chunking(chunking)

    web_scraper.set_chunking(chunking)

    youtube_extractor.set_chunking(chunking)





def make_progress_callback(job_manager: IngestJobManager, job_id: str):

    """Build on_progress handler for ingest_chunks."""



    def on_progress(phase: str, done: int, total: int) -> None:

        total = max(total, 1)

        if phase == "embedding":

            pct = 25 + int(65 * done / total)

            job_manager._update(

                job_id,

                status=JobStatus.EMBEDDING,

                message=f"Embedding {done}/{total} chunks...",

                progress=pct,

                chunks_total=total,

                chunks_done=done,

            )

        elif phase == "indexing":

            job_manager._update(

                job_id,

                status=JobStatus.INDEXING,

                message="Saving to index...",

                progress=95,

                chunks_total=total,

                chunks_done=done,

            )



    return on_progress





def ingest_chunks(

    chunks: List,

    notebook_id: int,

    source_name: str,

    embedding_generator,

    vector_db,

    memory,

    source_info: Dict[str, Any],

    replace_existing: bool = True,

    on_progress: Optional[Callable[[str, int, int], None]] = None,

    job_manager: Optional[IngestJobManager] = None,

    job_id: Optional[str] = None,

) -> int:

    """Embed chunks in batches, insert to Milvus incrementally. Returns source_id."""

    if replace_existing:

        vector_db.delete_by_source(source_name, notebook_id=notebook_id)

        existing = memory.find_source_by_name(source_name, notebook_id)

        if existing:

            memory.delete_source(existing["id"])



    total = len(chunks)

    if total == 0:

        source_info = dict(source_info)

        source_info["index_status"] = "ready"

        return memory.save_source(source_info, notebook_id=notebook_id)



    batch_size = getattr(embedding_generator, "batch_size", 32)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        offset = i

        def batch_callback(done: int, _batch_total: int, batch_offset: int = offset) -> None:
            if on_progress:
                on_progress("embedding", batch_offset + done, total)

        embedded_batch = embedding_generator.generate_embeddings(
            batch,
            batch_size=len(batch),
            on_batch=batch_callback,
        )
        vector_db.insert_embeddings(embedded_batch, notebook_id=notebook_id)



    if on_progress:

        on_progress("indexing", total, total)



    source_info = dict(source_info)

    source_info["index_status"] = "ready"

    source_id = memory.save_source(source_info, notebook_id=notebook_id)



    try:

        from src.generation.notebook_bm25 import notebook_bm25_cache

        notebook_bm25_cache.invalidate(notebook_id)

    except Exception:

        pass



    return source_id





def check_existing_checksum(memory, checksum: str, notebook_id: int, source_name: str) -> Optional[Dict[str, Any]]:

    """Return existing source if same file already indexed in notebook."""

    match = memory.find_source_by_checksum(checksum, notebook_id)

    if not match:

        return None

    if match.get("name") == source_name:

        return match

    return None


