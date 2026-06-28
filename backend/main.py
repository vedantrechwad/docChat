"""
DocChat — FastAPI Backend

Multi-notebook document Q&A with notes, AI writing assist, and export.
"""

import os
import re
import json
import time
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.background import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Global State ──────────────────────────────────────────────────────────────

_llm_router = None
_doc_processor = None
_embedding_generator = None
_vector_db = None
_rag_generator = None
_web_scraper = None
_youtube_extractor = None
_memory = None
_ingest_jobs = None
_orpheus_tts = None


def _initialize():
    """Initialize all components."""
    global _llm_router, _doc_processor, _embedding_generator
    global _vector_db, _rag_generator, _web_scraper, _youtube_extractor, _memory
    global _ingest_jobs, _orpheus_tts

    from src.llm.llm_router import LLMRouter
    from src.document_processing.doc_processor import DocumentProcessor
    from src.embeddings.embedding_generator import EmbeddingGenerator
    from src.vector_database.milvus_vector_db import MilvusVectorDB
    from src.generation.rag_v2 import RAGGeneratorV2
    from src.web_scraping.local_scraper import WebScraper
    from src.youtube.transcript import YouTubeTranscriptExtractor
    from src.memory.local_memory import LocalMemoryLayer
    from src.ingest.ingest_jobs import IngestJobManager
    from src.ingest.pipeline import apply_chunking_to_processors
    from backend.helpers import resolve_chunking

    Path("./data").mkdir(exist_ok=True)
    Path("./data/sources").mkdir(exist_ok=True)

    _llm_router = LLMRouter(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
        auto_start=True,
    )
    _memory = LocalMemoryLayer(db_path="./data/memory.db")
    _doc_processor = DocumentProcessor()
    _embedding_generator = EmbeddingGenerator()
    _vector_db = MilvusVectorDB(
        db_path="./data/docchat.db",
        collection_name="docchat",
        embedding_dim=_embedding_generator.get_embedding_dimension(),
    )
    _web_scraper = WebScraper()
    _youtube_extractor = YouTubeTranscriptExtractor()
    _ingest_jobs = IngestJobManager()
    # TTS lazy-init on first use (faster startup)

    chunking = resolve_chunking(_memory, _llm_router, notebook_id=1)
    apply_chunking_to_processors(chunking, _doc_processor, _web_scraper, _youtube_extractor)

    _rag_generator = RAGGeneratorV2(
        llm_router=_llm_router,
        embedding_generator=_embedding_generator,
        vector_db=_vector_db,
        memory=_memory,
    )

    try:
        _vector_db.create_index(use_binary_quantization=False)
    except Exception as e:
        logger.warning(f"Index creation skipped or failed: {e}")

    logger.info("CarnetLM initialized successfully")


def _shutdown():
    if _llm_router:
        _llm_router.close()
    if _vector_db:
        _vector_db.close()
    if _memory:
        _memory.close()
    if _orpheus_tts:
        _orpheus_tts.close()


def _get_tts():
    """Lazy-init Orpheus TTS on first use."""
    global _orpheus_tts
    if _orpheus_tts is None:
        from src.tts.orpheus_client import OrpheusTTSClient
        _orpheus_tts = OrpheusTTSClient()
    return _orpheus_tts


def _conversation_turns() -> int:
    if _memory and _memory.get_performance_mode() == "fast":
        return 3
    return 5


def _require_quality_mode(feature: str):
    if _memory and _memory.get_performance_mode() == "fast":
        raise HTTPException(
            status_code=403,
            detail=f"{feature} is available in Quality mode only. Switch mode in the sidebar.",
        )


def _apply_notebook_chunking(notebook_id: int, preset_override: Optional[str] = None):
    from backend.helpers import resolve_chunking
    from src.ingest.pipeline import apply_chunking_to_processors
    chunking = resolve_chunking(_memory, _llm_router, notebook_id, preset_override)
    apply_chunking_to_processors(chunking, _doc_processor, _web_scraper, _youtube_extractor)
    return chunking


# ─── Pydantic Models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    notebook_id: int = 1

class ChatResponse(BaseModel):
    response: str
    sources_used: list
    retrieval_count: int

class URLRequest(BaseModel):
    urls: List[str]
    notebook_id: int = 1

class YouTubeRequest(BaseModel):
    url: str
    notebook_id: int = 1

class NotebookCreate(BaseModel):
    name: str
    is_private: Optional[int] = 0
    password_hash: Optional[str] = None

class NotebookRename(BaseModel):
    name: str

class NotebookVerify(BaseModel):
    password_hash: str

class NoteCreate(BaseModel):
    title: str
    content: str = ""

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

class NoteAppend(BaseModel):
    text: str

class ModelSelect(BaseModel):
    model: str

class AIAssistRequest(BaseModel):
    text: str
    action: str  # grammar, rewrite, define, simplify, expand

class ExportRequest(BaseModel):
    html: str
    format: str  # pdf, docx, txt, md
    filename: str = "document"

class SummaryRequest(BaseModel):
    notebook_id: int = 1

class CompareRequest(BaseModel):
    source_names: List[str]
    notebook_id: int = 1

class BatchURLRequest(BaseModel):
    urls: List[str]
    notebook_id: int = 1

class ClipboardRequest(BaseModel):
    text: str
    title: str = "Pasted Text"
    notebook_id: int = 1

class SearchRequest(BaseModel):
    query: str
    notebook_id: int = 1

class ChatExportRequest(BaseModel):
    notebook_id: int = 1
    format: str = "md"  # md, txt

class RefreshSourceRequest(BaseModel):
    source_id: int
    notebook_id: int = 1

class ChunkingSettingsUpdate(BaseModel):
    preset: str = "auto"
    chunk_tokens: int = 384
    overlap_tokens: int = 100
    ingest_mode: str = "quality"
    notebook_id: int = 1

class DiscoverSettingsUpdate(BaseModel):
    enabled: bool = False
    auto_on_topic: bool = False
    max_results: int = 8
    provider: str = "duckduckgo"
    api_key: str = ""
    notebook_id: int = 1

class DiscoverSearchRequest(BaseModel):
    notebook_id: int = 1
    query: Optional[str] = None

class DiscoverIngestRequest(BaseModel):
    notebook_id: int = 1
    urls: List[str]

class SourceContentUpdate(BaseModel):
    content: str
    page_text: Optional[dict] = None

class DocumentSave(BaseModel):
    html_content: str
    title: str = "Untitled"

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: float = 1.0

class NoteIndexRequest(BaseModel):
    indexed: bool = True

class PerformanceSettingsUpdate(BaseModel):
    mode: str = "fast"





# ─── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _initialize()
    yield
    _shutdown()

app = FastAPI(
    title="CarnetLM",
    description="Multi-notebook document Q&A with AI writing assist.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    milvus_ok = bool(_vector_db and _vector_db.collection_exists)
    memory_ok = bool(_memory)
    tts_status = _orpheus_tts.health_check() if _orpheus_tts else {"available": False}
    llm_health = _llm_router.health_check() if _llm_router else {}
    ollama_models = _llm_router.list_models() if _llm_router and _llm_router.ollama_available else []
    return {
        "status": "ok" if _llm_router and milvus_ok and memory_ok else "degraded",
        "llm": llm_health,
        "ollama_has_models": len(ollama_models) > 0,
        "active_provider": _llm_router.get_active_provider() if _llm_router else "none",
        "performance_mode": _memory.get_performance_mode() if _memory else "fast",
        "milvus": {"available": milvus_ok},
        "memory": {"available": memory_ok},
        "tts": tts_status,
    }


# ─── Models ────────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """List available models (Ollama + Gemini)."""
    if not _llm_router:
        raise HTTPException(status_code=503, detail="Not initialized")
    models = _llm_router.list_models()
    # Include Gemini as a model option
    gemini_entry = None
    if _llm_router.gemini_available:
        gemini_entry = {
            "name": "gemini-2.5-flash",
            "size": "API",
            "family": "Gemini",
            "parameters": "",
            "quantization": "",
            "active": _llm_router.get_active_provider() == "gemini",
            "is_api": True,
        }
    return {
        "models": models,
        "gemini": gemini_entry,
        "active": _llm_router.ollama_model,
        "active_provider": _llm_router.get_active_provider(),
        "ollama_available": _llm_router.ollama_available,
        "gemini_available": _llm_router.gemini_available,
        "context_size": _llm_router.get_model_context_size(),
    }

@app.post("/api/models/select")
async def select_model(request: ModelSelect):
    """Switch the active model."""
    if not _llm_router:
        raise HTTPException(status_code=503, detail="Not initialized")
    if _llm_router.set_model(request.model):
        return {
            "status": "ok",
            "model": request.model,
            "active_provider": _llm_router.get_active_provider(),
            "context_size": _llm_router.get_model_context_size(),
        }
    raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")


# ─── Chunking Settings ───────────────────────────────────────────────────────

@app.get("/api/chunking/profiles")
async def chunking_profiles(notebook_id: int = Query(1)):
    from backend.helpers import get_chunking_profiles
    if not _llm_router or not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    profiles = get_chunking_profiles(_llm_router)
    profiles["current"] = _memory.get_chunking_settings(notebook_id)
    return profiles


@app.get("/api/settings/chunking")
async def get_chunking_settings(notebook_id: int = Query(1)):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    return _memory.get_chunking_settings(notebook_id)


@app.put("/api/settings/chunking")
async def update_chunking_settings(request: ChunkingSettingsUpdate):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    settings = {
        "preset": request.preset,
        "chunk_tokens": request.chunk_tokens,
        "overlap_tokens": request.overlap_tokens,
        "ingest_mode": request.ingest_mode,
    }
    _memory.set_chunking_settings(settings, notebook_id=request.notebook_id)
    _apply_notebook_chunking(request.notebook_id)
    return {"status": "ok", "settings": settings}


@app.get("/api/settings/performance")
async def get_performance_settings():
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    return {"mode": _memory.get_performance_mode()}


@app.put("/api/settings/performance")
async def update_performance_settings(request: PerformanceSettingsUpdate):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    _memory.set_performance_mode(request.mode)
    return {"status": "ok", "mode": _memory.get_performance_mode()}


@app.get("/api/settings/discover")
async def get_discover_settings(notebook_id: int = Query(1)):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    return _memory.get_discover_settings(notebook_id)


@app.put("/api/settings/discover")
async def update_discover_settings(request: DiscoverSettingsUpdate):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    settings = {
        "enabled": request.enabled,
        "auto_on_topic": request.auto_on_topic,
        "max_results": request.max_results,
        "provider": request.provider,
        "api_key": request.api_key,
    }
    _memory.set_discover_settings(settings, notebook_id=request.notebook_id)
    return {"status": "ok", "settings": settings}


@app.post("/api/discover/search")
async def discover_search(request: DiscoverSearchRequest):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    from src.discovery.web_discovery import infer_topic, search_candidates

    settings = _memory.get_discover_settings(request.notebook_id)
    if not settings.get("enabled"):
        raise HTTPException(status_code=403, detail="Web discovery is disabled in Settings")

    notebooks = _memory.list_notebooks()
    nb_name = next((n["name"] for n in notebooks if n["id"] == request.notebook_id), "Notebook")
    source_titles = [s.get("name", "") for s in _memory.get_sources(request.notebook_id)]
    query = (request.query or "").strip() or infer_topic(nb_name, source_titles)

    candidates = search_candidates(
        query=query,
        provider=settings.get("provider", "duckduckgo"),
        max_results=int(settings.get("max_results", 8)),
        api_key=settings.get("api_key") or None,
    )
    return {"query": query, "candidates": candidates}


@app.post("/api/discover/ingest")
async def discover_ingest(request: DiscoverIngestRequest):
    """Ingest user-selected URLs from discovery."""
    if not _web_scraper or not _ingest_jobs:
        raise HTTPException(status_code=503, detail="Not initialized")
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs selected")

    url_request = URLRequest(urls=request.urls, notebook_id=request.notebook_id)
    return await add_urls(url_request)


# ─── Ingest Jobs ─────────────────────────────────────────────────────────────

@app.get("/api/ingest/jobs")
async def list_ingest_jobs(notebook_id: Optional[int] = Query(None)):
    if not _ingest_jobs:
        return {"jobs": []}
    return {"jobs": _ingest_jobs.list_jobs(notebook_id=notebook_id)}


@app.get("/api/ingest/jobs/{job_id}")
async def get_ingest_job(job_id: str):
    if not _ingest_jobs:
        raise HTTPException(status_code=503, detail="Not initialized")
    job = _ingest_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


# ─── Notebooks ─────────────────────────────────────────────────────────────────

@app.get("/api/notebooks")
async def list_notebooks():
    if not _memory:
        return {"notebooks": []}
    return {"notebooks": _memory.list_notebooks()}

@app.post("/api/notebooks")
async def create_notebook(request: NotebookCreate):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    nb_id = _memory.create_notebook(
        name=request.name,
        is_private=request.is_private,
        password_hash=request.password_hash
    )
    return {"id": nb_id, "name": request.name, "status": "ok"}

@app.post("/api/notebooks/{notebook_id}/verify")
async def verify_notebook_password(notebook_id: int, request: NotebookVerify):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    success = _memory.verify_notebook_password(notebook_id, request.password_hash)
    return {"success": success}

@app.put("/api/notebooks/{notebook_id}")
async def rename_notebook(notebook_id: int, request: NotebookRename):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    if _memory.rename_notebook(notebook_id, request.name):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Notebook not found")

@app.delete("/api/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: int, password_hash: Optional[str] = Query(None)):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    # If notebook is private, check password hash
    notebooks = _memory.list_notebooks()
    target = next((n for n in notebooks if n["id"] == notebook_id), None)
    if target and target.get("is_private"):
        if not password_hash or target.get("password_hash") != password_hash:
            raise HTTPException(status_code=403, detail="Invalid password for private notebook")

    if _ingest_jobs:
        _ingest_jobs.cancel_notebook_jobs(notebook_id)
    if _memory.delete_notebook(notebook_id):
        if _vector_db:
            _vector_db.delete_by_notebook(notebook_id)
        _memory.delete_chunk_texts_by_notebook(notebook_id)
        from src.generation.notebook_bm25 import notebook_bm25_cache
        from src.ingest.pipeline import delete_notebook_source_files
        notebook_bm25_cache.invalidate(notebook_id)
        delete_notebook_source_files(notebook_id)
        return {"status": "ok"}
    raise HTTPException(status_code=400, detail="Cannot delete the last notebook")


# ─── Sources ──────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    notebook_id: int = Form(1),
    chunk_preset: str = Form("auto"),
):
    """Upload and process document files (background ingest)."""
    if not _doc_processor or not _ingest_jobs:
        raise HTTPException(status_code=503, detail="Not initialized")

    from src.ingest.ingest_jobs import JobStatus
    from src.ingest.pipeline import (
        check_existing_checksum,
        file_checksum,
        ingest_chunks,
        make_progress_callback,
        store_source_file,
    )

    _apply_notebook_chunking(notebook_id, chunk_preset if chunk_preset != "auto" else None)
    job_ids = []

    for uploaded_file in files:
        content = await uploaded_file.read()
        filename = uploaded_file.filename or "upload"
        suffix = Path(filename).suffix
        job = _ingest_jobs.create_job(notebook_id, filename)
        job_ids.append(job.id)
        jid = job.id
        nb_id = notebook_id

        def process_upload(file_content: bytes, fname: str, ext: str, job_id: str, notebook: int):
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file_content)
                    temp_path = tmp.name

                checksum = file_checksum(Path(temp_path))
                existing = check_existing_checksum(_memory, checksum, notebook, fname)
                if existing:
                    _ingest_jobs._update(
                        job_id, status=JobStatus.COMPLETED, progress=100,
                        message="Already indexed (unchanged file)",
                        chunks_total=existing.get("chunks", 0),
                        chunks_done=existing.get("chunks", 0),
                    )
                    return {
                        "file": fname, "status": "skipped", "message": "Already indexed",
                        "source_id": existing["id"], "chunks": existing.get("chunks", 0),
                    }
                
                # Note: If file exists in other notebooks, we still proceed to add it to this notebook
                # The chunks will be reused via deduplication, but the source record will be created
                _ingest_jobs._update(job_id, status=JobStatus.EXTRACTING, message="Extracting...", progress=10)
                
                # Check chunk cache first to avoid re-chunking
                from src.document_processing.chunk_cache import get_chunk_cache
                chunk_cache = get_chunk_cache()
                cached_chunks = chunk_cache.get_chunks(checksum)
                
                if cached_chunks:
                    chunks = cached_chunks
                    page_text = None  # Page text not cached for simplicity
                    logger.info(f"Retrieved {len(chunks)} chunks from cache for {fname}")
                    _ingest_jobs._update(
                        job_id, status=JobStatus.EMBEDDING,
                        message=f"Using cached chunks ({len(chunks)})",
                        progress=20, chunks_total=len(chunks), chunks_done=0,
                    )
                else:
                    result = _doc_processor.process_document(temp_path)
                    chunks = result.chunks
                    page_text = result.page_text
                    for chunk in chunks:
                        chunk.source_file = fname

                if not chunks:
                    raise ValueError(
                        "No content extracted from this file. "
                        "The PDF may be image-only, or the file may be empty."
                    )

                _ingest_jobs._update(
                    job_id, status=JobStatus.EMBEDDING,
                    message=f"Embedding 0/{len(chunks)} chunks...",
                    progress=25, chunks_total=len(chunks), chunks_done=0,
                )

                on_progress = make_progress_callback(_ingest_jobs, job_id)
                source_id = ingest_chunks(
                    chunks=chunks,
                    notebook_id=notebook,
                    source_name=fname,
                    embedding_generator=_embedding_generator,
                    vector_db=_vector_db,
                    memory=_memory,
                    source_info={
                        "name": fname,
                        "type": "Document",
                        "size": f"{len(file_content) / 1024:.1f} KB",
                        "chunks": len(chunks),
                        "checksum": checksum,
                        "index_status": "ready",
                    },
                    replace_existing=True,
                    on_progress=on_progress,
                    job_manager=_ingest_jobs,
                    job_id=job_id,
                    checksum=checksum,
                )

                stored = store_source_file(notebook, source_id, Path(temp_path), fname)
                _memory.save_source_file(
                    source_id, notebook, str(stored),
                    mime_type=ext, checksum=checksum, page_text=page_text,
                )

                return {"file": fname, "chunks": len(chunks), "status": "ok", "source_id": source_id}
            finally:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

        _ingest_jobs.enqueue(
            jid,
            lambda fc=content, fn=filename, ex=suffix, j=jid, n=nb_id: process_upload(fc, fn, ex, j, n),
        )

    return {"job_ids": job_ids, "status": "processing"}


@app.post("/api/url")
async def add_urls(request: URLRequest):
    """Scrape web URLs and add as sources (background ingest)."""
    if not _web_scraper or not _ingest_jobs:
        raise HTTPException(status_code=503, detail="Not initialized")

    from src.ingest.ingest_jobs import JobStatus
    from src.ingest.pipeline import ingest_chunks, make_progress_callback, check_existing_checksum

    _apply_notebook_chunking(request.notebook_id)
    job_ids = []

    for url in request.urls:
        job = _ingest_jobs.create_job(request.notebook_id, url)
        job_ids.append(job.id)
        jid = job.id
        nb_id = request.notebook_id

        def process_url(target_url: str, job_id: str, notebook: int):
            # Calculate checksum from URL for deduplication
            checksum = hashlib.md5(target_url.encode()).hexdigest()
            
            # Check if already exists in this notebook
            existing = check_existing_checksum(_memory, checksum, notebook, target_url)
            if existing:
                _ingest_jobs._update(
                    job_id, status=JobStatus.COMPLETED, progress=100,
                    message="Already indexed (unchanged URL)",
                    chunks_total=existing.get("chunks", 0),
                    chunks_done=existing.get("chunks", 0),
                )
                return {
                    "url": target_url, "title": existing.get("title", target_url),
                    "chunks": existing.get("chunks", 0), "status": "skipped",
                    "source_id": existing["id"],
                }
            
            # Note: If URL exists in other notebooks, we still proceed to add it to this notebook
            # The chunks will be reused via deduplication, but the source record will be created
            _ingest_jobs._update(job_id, status=JobStatus.EXTRACTING, message="Scraping...", progress=20)
            
            # Check chunk cache first
            from src.document_processing.chunk_cache import get_chunk_cache
            chunk_cache = get_chunk_cache()
            cached_chunks = chunk_cache.get_chunks(checksum)
            
            if cached_chunks:
                chunks = cached_chunks
                logger.info(f"Retrieved {len(chunks)} chunks from cache for URL {target_url}")
                source_name = chunks[0].source_file
                page_title = (chunks[0].metadata or {}).get("title", source_name)
            else:
                chunks = _web_scraper.scrape_url(target_url)
                if not chunks:
                    raise ValueError("No content extracted from URL")
                source_name = chunks[0].source_file
                page_title = (chunks[0].metadata or {}).get("title", source_name)
                # Cache chunks for future reuse
                chunking_config = {"source_type": "Website", "url": target_url}
                chunk_cache.store_chunks(checksum, chunks, source_name, chunking_config)
            
            _ingest_jobs._update(
                job_id, status=JobStatus.EMBEDDING,
                message=f"Embedding 0/{len(chunks)} chunks...", progress=25,
                chunks_total=len(chunks), chunks_done=0,
            )
            on_progress = make_progress_callback(_ingest_jobs, job_id)
            ingest_chunks(
                chunks=chunks, notebook_id=notebook, source_name=source_name,
                embedding_generator=_embedding_generator, vector_db=_vector_db, memory=_memory,
                source_info={
                    "name": source_name, "type": "Website",
                    "title": page_title,
                    "size": f"{len(chunks)} chunks", "chunks": len(chunks), "url": target_url,
                    "checksum": checksum,
                    "index_status": "ready",
                },
                replace_existing=True,
                on_progress=on_progress,
                job_manager=_ingest_jobs,
                job_id=job_id,
                checksum=checksum,
            )
            return {"url": target_url, "title": page_title, "chunks": len(chunks), "status": "ok"}

        _ingest_jobs.run_in_background(
            jid, lambda u=url, j=jid, n=nb_id: process_url(u, j, n),
        )

    return {"job_ids": job_ids, "status": "processing"}


@app.post("/api/youtube")
async def add_youtube(request: YouTubeRequest):
    """Extract YouTube transcript and add as source (background ingest)."""
    if not _youtube_extractor or not _ingest_jobs:
        raise HTTPException(status_code=503, detail="Not initialized")

    from src.ingest.ingest_jobs import JobStatus
    from src.ingest.pipeline import ingest_chunks, make_progress_callback, check_existing_checksum

    _apply_notebook_chunking(request.notebook_id)
    job = _ingest_jobs.create_job(request.notebook_id, request.url)
    jid = job.id
    nb_id = request.notebook_id

    def process_yt(target_url: str, job_id: str, notebook: int):
        # Calculate checksum from URL for deduplication
        checksum = hashlib.md5(target_url.encode()).hexdigest()
        
        # Check if already exists in this notebook
        existing = check_existing_checksum(_memory, checksum, notebook, target_url)
        if existing:
            _ingest_jobs._update(
                job_id, status=JobStatus.COMPLETED, progress=100,
                message="Already indexed (unchanged video)",
                chunks_total=existing.get("chunks", 0),
                chunks_done=existing.get("chunks", 0),
            )
            return {
                "title": existing.get("name", target_url), "chunks": existing.get("chunks", 0),
                "status": "skipped", "source_id": existing["id"],
            }
        
        # Note: If video exists in other notebooks, we still proceed to add it to this notebook
        # The chunks will be reused via deduplication, but the source record will be created
        _ingest_jobs._update(job_id, status=JobStatus.EXTRACTING, message="Extracting transcript...", progress=20)
        
        # Check chunk cache first
        from src.document_processing.chunk_cache import get_chunk_cache
        chunk_cache = get_chunk_cache()
        cached_chunks = chunk_cache.get_chunks(checksum)
        
        if cached_chunks:
            chunks = cached_chunks
            logger.info(f"Retrieved {len(chunks)} chunks from cache for YouTube video {target_url}")
            source_name = chunks[0].source_file
        else:
            chunks = _youtube_extractor.extract_transcript(target_url)
            if not chunks:
                raise ValueError("No transcript found")
            source_name = chunks[0].source_file
            # Cache chunks for future reuse
            chunking_config = {"source_type": "YouTube", "url": target_url}
            chunk_cache.store_chunks(checksum, chunks, source_name, chunking_config)
        
        _ingest_jobs._update(
            job_id, status=JobStatus.EMBEDDING,
            message=f"Embedding 0/{len(chunks)} chunks...", progress=25,
            chunks_total=len(chunks), chunks_done=0,
        )
        on_progress = make_progress_callback(_ingest_jobs, job_id)
        ingest_chunks(
            chunks=chunks, notebook_id=notebook, source_name=source_name,
            embedding_generator=_embedding_generator, vector_db=_vector_db, memory=_memory,
            source_info={
                "name": source_name, "type": "YouTube",
                "size": f"{len(chunks)} chunks", "chunks": len(chunks), "url": target_url,
                "checksum": checksum,
                "index_status": "ready",
            },
            replace_existing=True,
            on_progress=on_progress,
            job_manager=_ingest_jobs,
            job_id=job_id,
            checksum=checksum,
        )
        return {"title": source_name, "chunks": len(chunks), "status": "ok"}

    _ingest_jobs.run_in_background(
        jid, lambda u=request.url, j=jid, n=nb_id: process_yt(u, j, n),
    )
    return {"job_ids": [jid], "status": "processing"}


@app.get("/api/sources")
async def get_sources(notebook_id: int = Query(1)):
    if not _memory:
        return {"sources": []}
    return {"sources": _memory.get_sources(notebook_id)}


@app.delete("/api/sources/{source_id}")
async def delete_source(source_id: int):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    source_info = _memory.get_source_by_id(source_id)
    if not source_info:
        raise HTTPException(status_code=404, detail="Source not found")

    notebook_id = source_info["notebook_id"]
    source_name = source_info["name"]

    if _ingest_jobs:
        _ingest_jobs.cancel_jobs_for_source(notebook_id, source_name)

    if _vector_db:
        _vector_db.delete_by_source(source_file=source_name, notebook_id=notebook_id)

    from src.generation.notebook_bm25 import notebook_bm25_cache
    from src.ingest.pipeline import delete_source_files

    _memory.delete_chunk_texts_by_source(source_id)
    if not _memory.delete_source(source_id):
        raise HTTPException(status_code=404, detail="Source not found")

    notebook_bm25_cache.invalidate(notebook_id)
    delete_source_files(notebook_id, source_id)
    return {"status": "ok"}


# ─── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about your sources."""
    if not _rag_generator:
        raise HTTPException(status_code=503, detail="Not initialized")

    conv_context = _memory.get_conversation_context(
        notebook_id=request.notebook_id, max_turns=_conversation_turns(),
    )

    result = _rag_generator.generate_response(
        query=request.query,
        notebook_id=request.notebook_id,
        conversation_context=conv_context,
    )

    _memory.save_conversation_turn(result, notebook_id=request.notebook_id)

    return ChatResponse(
        response=result.response,
        sources_used=result.sources_used,
        retrieval_count=result.retrieval_count,
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream RAG response via Server-Sent Events."""
    if not _rag_generator or not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    # Auto-discover: fetch relevant websites if enabled
    discover_settings = _memory.get_discover_settings(request.notebook_id)
    if discover_settings.get("enabled") and discover_settings.get("auto_on_topic"):
        try:
            from src.discovery.web_discovery import search_candidates
            candidates = search_candidates(
                query=request.query,
                provider=discover_settings.get("provider", "duckduckgo"),
                max_results=3,
                api_key=discover_settings.get("api_key") or None,
            )
            if candidates:
                # Auto-ingest top 2 results in background
                top_urls = [c["url"] for c in candidates[:2]]
                url_request = URLRequest(urls=top_urls, notebook_id=request.notebook_id)
                # Trigger background ingest without blocking chat
                import asyncio
                asyncio.create_task(add_urls(url_request))
        except Exception as e:
            logger.warning(f"Auto-discover failed: {e}")

    conv_context = _memory.get_conversation_context(
        notebook_id=request.notebook_id, max_turns=_conversation_turns(),
    )

    def event_generator():
        full_text = []
        sources_used = []
        retrieval_count = 0
        try:
            for event_type, data in _rag_generator.generate_response_stream(
                query=request.query,
                notebook_id=request.notebook_id,
                conversation_context=conv_context,
            ):
                if event_type == "status":
                    yield f"event: status\ndata: {json.dumps(data)}\n\n"
                elif event_type == "token":
                    full_text.append(data.get("text", ""))
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                elif event_type == "meta":
                    sources_used = data.get("sources_used", [])
                    retrieval_count = data.get("retrieval_count", 0)
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                elif event_type == "done":
                    sources_used = data.get("sources_used", sources_used)
                    retrieval_count = data.get("retrieval_count", retrieval_count)
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                else:
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

            if full_text:
                from src.generation.rag_v2 import RAGResult
                result = RAGResult(
                    query=request.query,
                    response="".join(full_text),
                    sources_used=sources_used,
                    retrieval_count=retrieval_count,
                )
                _memory.save_conversation_turn(result, notebook_id=request.notebook_id)
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/history")
async def get_history(notebook_id: int = Query(1)):
    if not _memory:
        return {"history": []}
    return {"history": _memory.get_chat_history(notebook_id)}


@app.delete("/api/history")
async def clear_history(notebook_id: int = Query(1)):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    _memory.clear_chat(notebook_id)
    return {"status": "ok"}


# NOTE: /api/chat/stream removed — was fake streaming (generated full response
# synchronously then split into 3-word chunks). The frontend now uses /api/chat.
# True streaming would require the LLM router to yield tokens.


# ─── Auto-Summary ──────────────────────────────────────────────────────────────

@app.get("/api/summary")
async def get_summary(notebook_id: int = Query(...)):
    """Fetch saved study guide/summary for a notebook."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    guide = _memory.get_study_guide(notebook_id)
    if guide:
        return json.loads(guide)
    return {"summary": "", "sources_used": []}


@app.post("/api/summary")
async def generate_summary(request: SummaryRequest):
    """Auto-generate and save a structured study guide with inline citations."""
    if not _llm_router or not _vector_db or not _embedding_generator or not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    # Get all sources for this notebook
    sources = _memory.get_sources(request.notebook_id)
    if not sources:
        raise HTTPException(status_code=400, detail="No sources available to summarize")

    # Retrieve a broad set of chunks across all sources
    query_vector = _embedding_generator.generate_query_embedding(
        "main topics themes key points summary overview"
    )
    search_results = _vector_db.search(
        query_vector=query_vector.tolist(),
        limit=20,
        notebook_id=request.notebook_id,
    )

    if not search_results:
        raise HTTPException(status_code=400, detail="No content found in sources")

    # Build context from retrieved chunks with clear reference numbers
    context_parts = []
    sources_used = []
    for i, r in enumerate(search_results[:15]):
        ref = f"[{i + 1}]"
        citation = r.get("citation", {})
        context_parts.append(f"{ref} {r['content']}")
        sources_used.append({
            "reference": ref,
            "source_file": citation.get("source_file", "Unknown"),
            "source_type": citation.get("source_type", "unknown"),
            "page_number": citation.get("page_number"),
            "chunk_id": r.get("id", ""),
            "chunk_index": citation.get("chunk_index"),
            "relevance_score": r.get("score", r.get("rrf_score", 0)),
            "text": r["content"][:300],
        })

    context = "\n\n".join(context_parts)

    try:
        result = _llm_router.generate(
            prompt=f"""Based on the following source material, create a comprehensive study guide.
Cite the sources you use using their numbers, e.g. [1], [2], etc.

Rules:
1. Every major fact, concept, or summary MUST be grounded in the context and cite the corresponding source numbers, e.g. [1].
2. Structure the guide with the following sections:
   - **Executive Summary**: A high-level overview of the notebook materials.
   - **Key Concepts**: Core terms, definitions, and theories, citing source numbers.
   - **Important Details**: Facts, figures, or notable connections.

Source Material:
{context}""",
            system_prompt="You are an expert study guide creator. Create structured summaries with clear source citations [1], [2], etc.",
            temperature=0.3,
            max_tokens=2000,
        )
        
        guide_data = {
            "summary": result.content,
            "sources_used": sources_used,
            "provider": result.provider,
            "model": result.model,
        }
        
        # Save to database
        _memory.save_study_guide(request.notebook_id, json.dumps(guide_data))
        return guide_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Source Content Viewer ─────────────────────────────────────────────────────

@app.get("/api/sources/{source_id}/content")
async def get_source_content(source_id: int, notebook_id: int = Query(1)):
    """Get the full content of a source (all its chunks)."""
    from backend.helpers import sort_chunks

    if not _vector_db or not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    source_info = _memory.get_source_by_id(source_id)
    if not source_info:
        raise HTTPException(status_code=404, detail="Source not found")

    nb_id = source_info["notebook_id"]
    results = _vector_db.query_by_source(
        source_file=source_info["name"],
        notebook_id=nb_id,
    )
    results = sort_chunks(results)

    sf = _memory.get_source_file(source_id)
    editable = source_info.get("type") in ("Document", "Clipboard") or bool(sf)

    return {
        "source": source_info,
        "chunks": [
            {
                "content": r["content"],
                "page_number": r["citation"].get("page_number"),
                "chunk_index": r["citation"].get("chunk_index", 0),
            }
            for r in results
        ],
        "total_chunks": len(results),
        "editable": editable,
        "page_text": sf.get("page_text", {}) if sf else {},
        "file_path": sf.get("file_path") if sf else None,
    }


@app.put("/api/sources/{source_id}/content")
async def update_source_content(source_id: int, request: SourceContentUpdate):
    """Save edited source content and re-index."""
    from backend.helpers import reindex_source_content

    if not _doc_processor or not _embedding_generator or not _vector_db or not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    source_info = _memory.get_source_by_id(source_id)
    if not source_info:
        raise HTTPException(status_code=404, detail="Source not found")

    sf = _memory.get_source_file(source_id)
    notebook_id = source_info["notebook_id"]
    _apply_notebook_chunking(notebook_id)

    if sf and sf.get("file_path"):
        path = Path(sf["file_path"])
        if request.page_text:
            _memory.save_source_file(
                source_id, notebook_id, sf["file_path"],
                mime_type=sf.get("mime_type", ""),
                checksum=sf.get("checksum", ""),
                page_text=request.page_text,
            )
        elif path.suffix.lower() in (".txt", ".md"):
            path.write_text(request.content, encoding="utf-8")

    chunk_count = reindex_source_content(
        source_id=source_id,
        content=request.content,
        page_text=request.page_text,
        doc_processor=_doc_processor,
        embedding_generator=_embedding_generator,
        vector_db=_vector_db,
        memory=_memory,
        source_info=source_info,
    )

    return {"status": "ok", "chunks": chunk_count, "message": "Source re-indexed"}


# ─── Full-Text Search ──────────────────────────────────────────────────────────

@app.post("/api/search")
async def full_text_search(request: SearchRequest):
    """Search across all source content using vector similarity."""
    if not _vector_db or not _embedding_generator:
        raise HTTPException(status_code=503, detail="Not initialized")

    query_vector = _embedding_generator.generate_query_embedding(request.query)
    results = _vector_db.search(
        query_vector=query_vector.tolist(),
        limit=20,
        notebook_id=request.notebook_id,
    )

    return {
        "results": [
            {
                "content": r["content"][:500],
                "source_file": r["citation"]["source_file"],
                "source_type": r["citation"]["source_type"],
                "page_number": r["citation"].get("page_number"),
                "score": r["score"],
                "chunk_id": r["id"],
            }
            for r in results
        ],
        "total": len(results),
    }


# ─── Chat Export ───────────────────────────────────────────────────────────────

@app.post("/api/chat/export")
async def export_chat(request: ChatExportRequest):
    """Export chat history as Markdown or plain text."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    history = _memory.get_chat_history(request.notebook_id)
    if not history:
        raise HTTPException(status_code=400, detail="No chat history to export")

    if request.format == "md":
        lines = ["# DocChat — Chat Export\n"]
        for msg in history:
            role = "**You**" if msg["role"] == "user" else "**DocChat**"
            lines.append(f"### {role}\n")
            lines.append(msg["content"] + "\n")
            if msg.get("sources") and msg["role"] == "assistant":
                src_names = list(set(s.get("source_file", "") for s in msg["sources"] if s.get("source_file")))
                if src_names:
                    lines.append(f"*Sources: {', '.join(src_names)}*\n")
            lines.append("---\n")
        content = "\n".join(lines)
        media_type = "text/markdown"
        ext = "md"
    else:
        lines = []
        for msg in history:
            role = "You" if msg["role"] == "user" else "DocChat"
            lines.append(f"{role}:")
            lines.append(msg["content"])
            lines.append("")
        content = "\n".join(lines)
        media_type = "text/plain"
        ext = "txt"

    return Response(
        content=content.encode("utf-8"),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="chat_export.{ext}"'},
    )


# ─── Compare Sources ──────────────────────────────────────────────────────────

@app.post("/api/compare")
async def compare_sources(request: CompareRequest):
    """Compare 2+ sources using AI analysis."""
    if not _llm_router or not _vector_db or not _embedding_generator:
        raise HTTPException(status_code=503, detail="Not initialized")

    if len(request.source_names) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 sources to compare")

    # Get chunks from each source and build unified reference context
    context_parts = []
    sources_used = []
    cid = 1
    
    for name in request.source_names:
        results = _vector_db.query_by_source(
            source_file=name,
            notebook_id=request.notebook_id,
            limit=10,
        )
        # Sort by chunk_index to get representative content in order
        results.sort(key=lambda x: (
            x.get("citation", {}).get("page_number") or 0,
            x.get("citation", {}).get("chunk_index", 0),
        ))
        
        for r in results[:5]:  # limit to top 5 representative chunks per source
            ref = f"[{cid}]"
            citation = r.get("citation", {})
            context_parts.append(f"{ref} (Source: {name}):\n{r['content']}")
            sources_used.append({
                "reference": ref,
                "source_file": name,
                "source_type": citation.get("source_type", "unknown"),
                "page_number": citation.get("page_number"),
                "chunk_id": r.get("id", ""),
                "chunk_index": citation.get("chunk_index"),
                "relevance_score": r.get("score", r.get("rrf_score", 0)),
                "text": r["content"][:300],
            })
            cid += 1

    context = "\n\n".join(context_parts)

    try:
        result = _llm_router.generate(
            prompt=f"""Compare and contrast the following sources.
Cite the sources you reference using their numbers, e.g. [1], [2], etc.

Rules:
1. Every comparison fact or unique detail MUST cite the corresponding source numbers, e.g. [1].
2. Structure the comparison with the following sections:
   - **Common Themes**: What topics/ideas appear across the sources.
   - **Key Differences**: Where sources diverge or provide unique information.
   - **Comparison Table**: A markdown table comparing key aspects.
   - **Synthesis**: Summary of the combined knowledge.

Source Material:
{context}""",
            system_prompt="You are an expert analyst. Create structured, clear comparisons, citing sources with [1], [2], etc.",
            temperature=0.3,
            max_tokens=2000,
        )
        return {
            "comparison": result.content,
            "sources_used": sources_used,
            "provider": result.provider,
            "model": result.model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Clipboard Paste as Source ────────────────────────────────────────────────

@app.post("/api/clipboard")
async def add_clipboard_source(request: ClipboardRequest):
    """Add pasted text directly as a source (background ingest)."""
    if not _doc_processor or not _ingest_jobs:
        raise HTTPException(status_code=503, detail="Not initialized")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    from src.ingest.ingest_jobs import JobStatus
    from src.ingest.pipeline import ingest_chunks, make_progress_callback, check_existing_checksum

    _apply_notebook_chunking(request.notebook_id)
    job = _ingest_jobs.create_job(request.notebook_id, request.title)
    jid = job.id
    nb_id = request.notebook_id
    text = request.text
    title = request.title

    def process_clipboard():
        # Calculate checksum from text for deduplication
        checksum = hashlib.md5(text.encode()).hexdigest()
        
        # Check if already exists in this notebook
        existing = check_existing_checksum(_memory, checksum, nb_id, title)
        if existing:
            _ingest_jobs._update(
                jid, status=JobStatus.COMPLETED, progress=100,
                message="Already indexed (unchanged text)",
                chunks_total=existing.get("chunks", 0),
                chunks_done=existing.get("chunks", 0),
            )
            return {
                "title": existing.get("name", title), "chunks": existing.get("chunks", 0),
                "status": "skipped", "source_id": existing["id"],
            }
        
        # Note: If text exists in other notebooks, we still proceed to add it to this notebook
        # The chunks will be reused via deduplication, but the source record will be created
        _ingest_jobs._update(jid, status=JobStatus.EXTRACTING, message="Processing...", progress=20)
        
        # Check chunk cache first
        from src.document_processing.chunk_cache import get_chunk_cache
        chunk_cache = get_chunk_cache()
        cached_chunks = chunk_cache.get_chunks(checksum)
        
        if cached_chunks:
            chunks = cached_chunks
            logger.info(f"Retrieved {len(chunks)} chunks from cache for clipboard text")
        else:
            chunks = _doc_processor.process_text_content(text=text, source_file=title, source_type="clipboard")
            if not chunks:
                raise ValueError("No content to process")
            # Cache chunks for future reuse
            chunking_config = {"source_type": "Clipboard"}
            chunk_cache.store_chunks(checksum, chunks, title, chunking_config)
        
        _ingest_jobs._update(
            jid, status=JobStatus.EMBEDDING,
            message=f"Embedding 0/{len(chunks)} chunks...", progress=25,
            chunks_total=len(chunks), chunks_done=0,
        )
        on_progress = make_progress_callback(_ingest_jobs, jid)
        ingest_chunks(
            chunks=chunks, notebook_id=nb_id, source_name=title,
            embedding_generator=_embedding_generator, vector_db=_vector_db, memory=_memory,
            source_info={
                "name": title, "type": "Clipboard",
                "size": f"{len(text)} chars", "chunks": len(chunks),
                "checksum": checksum,
                "index_status": "ready",
            },
            replace_existing=True,
            on_progress=on_progress,
            job_manager=_ingest_jobs,
            job_id=jid,
            checksum=checksum,
        )
        return {"title": title, "chunks": len(chunks), "status": "ok"}

    _ingest_jobs.run_in_background(jid, process_clipboard)
    return {"job_ids": [jid], "status": "processing"}


# ─── Source Refresh ───────────────────────────────────────────────────────────

@app.post("/api/sources/refresh")
async def refresh_source(request: RefreshSourceRequest):
    """Re-scrape a web URL or re-extract a YouTube transcript."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    source_info = _memory.get_source_by_id(request.source_id)
    if not source_info:
        raise HTTPException(status_code=404, detail="Source not found")

    source_type = source_info.get("type", "")

    # Can only refresh web and YouTube sources
    if source_type not in ("Website", "YouTube"):
        raise HTTPException(status_code=400, detail="Can only refresh Website and YouTube sources")

    # Get the URL from source metadata (using proper API, not raw cursor)
    metadata = source_info.get("metadata", {})
    source_name = source_info["name"]
    nb_id = source_info["notebook_id"]

    if _vector_db:
        _vector_db.delete_by_source(source_name, notebook_id=nb_id)

    try:
        _apply_notebook_chunking(nb_id)
        if source_type == "Website":
            # Re-scrape
            url = metadata.get("url") or metadata.get("name", "")
            if not url.startswith("http"):
                raise HTTPException(status_code=400, detail="Cannot determine URL for this source")
            chunks = _web_scraper.scrape_url(url)
        elif source_type == "YouTube":
            url = metadata.get("url") or metadata.get("video_url", "")
            if not url:
                raise HTTPException(status_code=400, detail="Cannot determine URL for this source")
            chunks = _youtube_extractor.extract_transcript(url)
        else:
            chunks = []

        if chunks:
            new_source_name = chunks[0].source_file
            embedded = _embedding_generator.generate_embeddings(chunks)
            _vector_db.insert_embeddings(embedded, notebook_id=nb_id)
            _memory.update_source(request.source_id, {
                "name": new_source_name,
                "type": source_type,
                "size": f"{len(chunks)} chunks",
                "chunks": len(chunks),
                "url": url,
            }, notebook_id=nb_id)
            return {"title": new_source_name, "chunks": len(chunks), "status": "refreshed"}

        raise HTTPException(status_code=400, detail="No content extracted on refresh")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Notes ─────────────────────────────────────────────────────────────────────

@app.get("/api/notebooks/{notebook_id}/notes")
async def list_notes(notebook_id: int):
    if not _memory:
        return {"notes": []}
    return {"notes": _memory.list_notes(notebook_id)}


@app.post("/api/notebooks/{notebook_id}/notes")
async def create_note(notebook_id: int, request: NoteCreate):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    note_id = _memory.create_note(notebook_id, request.title, request.content)
    return {"id": note_id, "status": "ok"}


@app.get("/api/notes/{note_id}")
async def get_note(note_id: int):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    note = _memory.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@app.put("/api/notes/{note_id}")
async def update_note(note_id: int, request: NoteUpdate):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    if _memory.update_note(note_id, title=request.title, content=request.content):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Note not found")


@app.post("/api/notes/{note_id}/append")
async def append_to_note(note_id: int, request: NoteAppend):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    if _memory.append_to_note(note_id, request.text):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Note not found")


@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: int):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    note = _memory.get_note(note_id)
    if note and _vector_db:
        _vector_db.delete_by_source(f"note:{note['title']} (#{note_id})", notebook_id=note["notebook_id"])
    if _memory.delete_note(note_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Note not found")


@app.post("/api/notes/{note_id}/index")
async def index_note(note_id: int, request: NoteIndexRequest):
    """Index or remove a note from RAG retrieval."""
    from backend.helpers import index_note_as_source

    if not _memory or not _embedding_generator or not _vector_db:
        raise HTTPException(status_code=503, detail="Not initialized")

    note = _memory.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    _memory.set_note_indexed(note_id, request.indexed)
    source_name = f"note:{note['title']} (#{note_id})"

    if request.indexed and note.get("content", "").strip():
        index_note_as_source(note, note["notebook_id"], _embedding_generator, _vector_db)
    elif _vector_db:
        _vector_db.delete_by_source(source_name, notebook_id=note["notebook_id"])

    return {"status": "ok", "indexed": request.indexed}


# ─── Documents (Quill persistence) ───────────────────────────────────────────

@app.get("/api/notebooks/{notebook_id}/document")
async def get_notebook_document(notebook_id: int):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    doc = _memory.get_document(notebook_id)
    return doc or {"notebook_id": notebook_id, "title": "Untitled", "html_content": ""}


@app.put("/api/notebooks/{notebook_id}/document")
async def save_notebook_document(notebook_id: int, request: DocumentSave):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    doc_id = _memory.save_document(notebook_id, request.html_content, request.title)
    return {"status": "ok", "id": doc_id}


# ─── TTS ─────────────────────────────────────────────────────────────────────

@app.get("/api/tts/health")
async def tts_health():
    tts = _get_tts()
    status = tts.health_check()
    status["voices"] = tts.list_voices() if status.get("available") else []
    return status


@app.post("/api/tts")
async def synthesize_speech(request: TTSRequest):
    try:
        tts = _get_tts()
        audio = tts.synthesize(request.text, voice=request.voice, speed=request.speed)
        return Response(content=audio, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"TTS unavailable: {e}")


# ─── AI Writing Assist ────────────────────────────────────────────────────────

AI_PROMPTS = {
    "grammar": "Fix all grammar, spelling, and punctuation errors in the following text. Return ONLY the corrected text, nothing else:\n\n",
    "rewrite": "Rewrite the following text to be clearer and more professional. Return ONLY the rewritten text, nothing else:\n\n",
    "define": "Define the following word or phrase in 1-2 clear sentences:\n\n",
    "simplify": "Simplify the following text to make it easier to understand. Use simpler words and shorter sentences. Return ONLY the simplified text:\n\n",
    "expand": "Expand on the following text with more detail and examples. Return ONLY the expanded text:\n\n",
}

@app.post("/api/ai/assist")
async def ai_assist(request: AIAssistRequest):
    """AI writing assistant — grammar, rewrite, define, simplify, expand."""
    if not _llm_router:
        raise HTTPException(status_code=503, detail="Not initialized")

    if request.action not in AI_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}. Use: {list(AI_PROMPTS.keys())}")

    prompt = AI_PROMPTS[request.action] + request.text

    try:
        result = _llm_router.generate(
            prompt=prompt,
            system_prompt="You are a helpful writing assistant. Be concise and direct.",
            temperature=0.3,
            max_tokens=1000,
        )
        return {
            "result": result.content,
            "action": request.action,
            "provider": result.provider,
            "model": result.model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Export ────────────────────────────────────────────────────────────────────

@app.post("/api/export")
async def export_document(request: ExportRequest):
    """Export document in various formats."""

    if request.format == "txt":
        # Strip HTML tags → plain text
        clean = re.sub(r"<[^>]+>", "", request.html)
        clean = re.sub(r"&nbsp;", " ", clean)
        clean = re.sub(r"&amp;", "&", clean)
        clean = re.sub(r"&lt;", "<", clean)
        clean = re.sub(r"&gt;", ">", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
        return Response(
            content=clean.encode("utf-8"),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{request.filename}.txt"'},
        )

    elif request.format == "md":
        try:
            from markdownify import markdownify
            md = markdownify(request.html, heading_style="ATX", strip=["img"])
            return Response(
                content=md.encode("utf-8"),
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{request.filename}.md"'},
            )
        except ImportError:
            # Fallback: basic HTML to text
            clean = re.sub(r"<[^>]+>", "", request.html)
            return Response(
                content=clean.encode("utf-8"),
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{request.filename}.md"'},
            )

    elif request.format == "docx":
        try:
            from docx import Document
            from htmldocx import HtmlToDocx

            doc = Document()
            parser = HtmlToDocx()
            parser.add_html_to_document(request.html, doc)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                doc.save(tmp.name)
                tmp_path = tmp.name

            def _cleanup_temp():
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            # Use BackgroundTasks to clean up temp file after response is sent
            background_tasks = BackgroundTasks()
            background_tasks.add_task(_cleanup_temp)

            return FileResponse(
                tmp_path,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                filename=f"{request.filename}.docx",
                background=background_tasks,
            )
        except ImportError:
            raise HTTPException(status_code=500, detail="DOCX export requires python-docx and htmldocx. Run: uv pip install python-docx htmldocx")

    elif request.format == "pdf":
        # Use browser print as primary PDF method (no extra deps)
        raise HTTPException(status_code=400, detail="Use your browser's Print → Save as PDF (Ctrl+P) for PDF export.")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}. Use: txt, md, docx, pdf")


# ─── Editable Concept Board ───────────────────────────────────────────────────

class ConceptSaveRequest(BaseModel):
    id: Optional[int] = None
    notebook_id: int
    title: str
    explanation: str
    links: List[str] = []
    x: Optional[int] = None
    y: Optional[int] = None

class ConceptGenerateRequest(BaseModel):
    notebook_id: int
    prompt: str

@app.post("/api/concepts/generate")
async def generate_concept_from_prompt(request: ConceptGenerateRequest):
    """Generate a specific concept card using user prompt and relevant documents context."""
    if not _memory or not _llm_router or not _vector_db or not _embedding_generator:
        raise HTTPException(status_code=503, detail="Not initialized")

    prompt_text = request.prompt.strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        # Retrieve context from vector db matching the prompt
        query_vector = _embedding_generator.generate_query_embedding(prompt_text)
        search_results = _vector_db.search(
            query_vector=query_vector.tolist(),
            limit=6,
            notebook_id=request.notebook_id
        )

        context = "\n".join([r["content"] for r in search_results])
        sources = list(set([r["source_file"] for r in search_results if r.get("source_file")]))

        llm_prompt = f"""You are an elite academic tutor. Your goal is to write a high-quality active recall study flashcard (front/back pair) for the topic/question: "{prompt_text}"

Grounding Source Material Context:
{context[:4000]}

Instructions:
1. FRONT (title): Write a precise, clear question or key term (1-6 words) to place on the FRONT of the card (e.g. "What is neuroplasticity?" or "The role of myelin").
2. BACK (explanation): Write a highly accurate, rigorous answer or definition (1-3 sentences) to place on the BACK of the card. Incorporate specific facts or mechanisms from the context if available.
3. Fallback: If context is sparse or topic is off-topic, utilize your general academic knowledge to formulate an accurate and informative Q&A card.

Format your response EXACTLY as a raw JSON object with keys "title" and "explanation":
{{
  "title": "Front content",
  "explanation": "Back content"
}}"""

        res = _llm_router.generate(
            prompt=llm_prompt,
            system_prompt="You are a JSON-only response writer. Output only raw JSON object with no quotes or markdown.",
            temperature=0.3
        )
        raw_json = res.content.strip()
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()

        data = json.loads(raw_json)
        title = data.get("title", "Core Concept").strip()
        explanation = data.get("explanation", "").strip()

        if not explanation:
            explanation = f"Concept relating to: {prompt_text}"

        # Shift order of all existing concepts to make room for the new one at the top-left
        _memory.shift_concepts_order(request.notebook_id)

        new_id = _memory.create_concept(
            notebook_id=request.notebook_id,
            title=title,
            explanation=explanation,
            links_json=json.dumps(sources),
            x=50,
            y=50,
            sort_order=0
        )

        return {
            "id": new_id,
            "title": title,
            "explanation": explanation,
            "links": sources,
            "x": 50,
            "y": 50,
            "sort_order": 0
        }
    except Exception as e:
        logger.error(f"Failed to generate concept from prompt: {e}")
        _memory.shift_concepts_order(request.notebook_id)
        new_id = _memory.create_concept(
            notebook_id=request.notebook_id,
            title="Generated Concept",
            explanation=f"Double click to edit. AI could not generate from prompt: {prompt_text}",
            links_json="[]",
            x=50,
            y=50,
            sort_order=0
        )
        return {
            "id": new_id,
            "title": "Generated Concept",
            "explanation": f"Double click to edit. AI could not generate from prompt: {prompt_text}",
            "links": [],
            "x": 50,
            "y": 50,
            "sort_order": 0
        }

@app.get("/api/concepts")
async def get_concepts(notebook_id: int = Query(...)):
    """Fetch all concepts for a notebook. If empty and never generated, auto-generate initial ones from sources."""
    if not _memory or not _llm_router or not _vector_db:
        raise HTTPException(status_code=503, detail="Not initialized")

    concepts = _memory.list_concepts(notebook_id)
    if not concepts and not _memory.has_generated_concepts(notebook_id):
        # Auto-generate initial concepts using LLM
        sources = _memory.get_sources(notebook_id)
        if sources:
            source_content = ""
            try:
                search_results = _vector_db.search(
                    query_vector=[0.0]*384,
                    limit=5,
                    notebook_id=notebook_id
                )
                source_content = "\n".join([r["content"] for r in search_results])
            except Exception:
                pass

            if not source_content.strip():
                source_content = "Study notebook documents and learning materials."

            prompt = f"""You are a concept mapping agent. Analyze the following source material and identify 3 distinct core concepts, key terms, or core principles.
For each concept, provide:
1. A concise concept title (1 to 4 words).
2. A brief, clear explanation (1 to 2 sentences) summarizing what it is.

Source Material:
{source_content[:3000]}

Format your response EXACTLY as a JSON array of objects, with no other text, commentary, or markdown formatting:
[
  {{"title": "Concept Name", "explanation": "Short 1-2 sentence definition."}}
]"""
            try:
                res = _llm_router.generate(
                    prompt=prompt,
                    system_prompt="You are a JSON-only response writer. Output only raw JSON array with no quotes or markdown.",
                    temperature=0.3
                )
                raw_json = res.content.strip()
                if "```json" in raw_json:
                    raw_json = raw_json.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_json:
                    raw_json = raw_json.split("```")[1].split("```")[0].strip()

                initial_concepts = json.loads(raw_json)
                for index, ic in enumerate(initial_concepts):
                    links = [sources[0]["name"]] if len(sources) > 0 else []
                    # Stagger coordinates horizontally: starts at 50, offsets by 350px per card
                    stagger_x = 50 + (index * 350)
                    stagger_y = 50
                    _memory.create_concept(
                        notebook_id=notebook_id,
                        title=ic.get("title", "Core Concept"),
                        explanation=ic.get("explanation", "Short explanation of the concept."),
                        links_json=json.dumps(links),
                        x=stagger_x,
                        y=stagger_y,
                        sort_order=index
                    )
                _memory.mark_concepts_generated(notebook_id)
            except Exception as e:
                logger.error(f"Failed to auto-generate concepts: {e}")
                _memory.create_concept(
                    notebook_id=notebook_id,
                    title="Key Mindset Theme",
                    explanation="Double click to edit and add your concept explanation here.",
                    links_json="[]",
                    x=50,
                    y=50,
                    sort_order=0
                )
                _memory.mark_concepts_generated(notebook_id)
            concepts = _memory.list_concepts(notebook_id)

    return {"concepts": concepts}

@app.post("/api/concepts")
async def save_concept(request: ConceptSaveRequest):
    """Create or update a concept card."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    links_json = json.dumps(request.links)
    if request.id is not None:
        updated = _memory.update_concept(
            concept_id=request.id,
            title=request.title,
            explanation=request.explanation,
            links_json=links_json,
            x=request.x,
            y=request.y
        )
        return {"status": "updated", "success": updated}
    else:
        new_id = _memory.create_concept(
            notebook_id=request.notebook_id,
            title=request.title,
            explanation=request.explanation,
            links_json=links_json,
            x=request.x or 100,
            y=request.y or 100
        )
        return {"status": "created", "id": new_id}

class ConceptReorderRequest(BaseModel):
    notebook_id: int
    concept_ids: List[int]

@app.post("/api/concepts/reorder")
async def reorder_concepts(request: ConceptReorderRequest):
    """Reorder concepts for a notebook."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    try:
        for index, cid in enumerate(request.concept_ids):
            _memory.update_concept_order(cid, index)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to reorder concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ConceptGradeRequest(BaseModel):
    concept_id: int
    grade: str  # "easy", "good", "hard"

@app.post("/api/concepts/grade")
async def grade_concept(request: ConceptGradeRequest):
    """Grade a flashcard's recall difficulty, updating its Leitner Box level."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    if request.grade not in ["easy", "good", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid grade value")
    try:
        new_box = _memory.grade_concept_card(request.concept_id, request.grade)
        return {"status": "success", "new_box": new_box}
    except Exception as e:
        logger.error(f"Failed to grade flashcard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/concepts/{concept_id}")
async def delete_concept(concept_id: int):
    """Delete a concept card."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    deleted = _memory.delete_concept(concept_id)
    return {"status": "deleted", "success": deleted}


# ─── Static Frontend ───────────────────────────────────────────────────────────

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def serve_frontend():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "CarnetLM API is running. Frontend not found at /static/index.html"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
