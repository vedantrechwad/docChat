"""
DocChat — FastAPI Backend

Upload documents, websites, or YouTube videos.
Ask questions. Get cited answers.
"""

import os
import time
import logging
import tempfile
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Global State ──────────────────────────────────────────────────────────────

# These are initialized once on startup and shared across requests.
_llm_router = None
_doc_processor = None
_embedding_generator = None
_vector_db = None
_rag_generator = None
_web_scraper = None
_youtube_extractor = None
_memory = None


def _initialize():
    """Initialize all components."""
    global _llm_router, _doc_processor, _embedding_generator
    global _vector_db, _rag_generator, _web_scraper, _youtube_extractor, _memory

    from src.llm.llm_router import LLMRouter
    from src.document_processing.doc_processor import DocumentProcessor
    from src.embeddings.embedding_generator import EmbeddingGenerator
    from src.vector_database.milvus_vector_db import MilvusVectorDB
    from src.generation.rag_v2 import RAGGeneratorV2
    from src.web_scraping.local_scraper import WebScraper
    from src.youtube.transcript import YouTubeTranscriptExtractor
    from src.memory.local_memory import LocalMemoryLayer

    # Ensure data directory exists
    Path("./data").mkdir(exist_ok=True)

    _llm_router = LLMRouter(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
    )
    _doc_processor = DocumentProcessor()
    _embedding_generator = EmbeddingGenerator()
    _vector_db = MilvusVectorDB(
        db_path="./data/docchat.db",
        collection_name="docchat",
    )
    _rag_generator = RAGGeneratorV2(
        llm_router=_llm_router,
        embedding_generator=_embedding_generator,
        vector_db=_vector_db,
    )
    _web_scraper = WebScraper()
    _youtube_extractor = YouTubeTranscriptExtractor()
    _memory = LocalMemoryLayer(db_path="./data/memory.db")

    # Create vector index
    try:
        _vector_db.create_index(use_binary_quantization=False)
    except Exception:
        pass  # Index may already exist

    logger.info("DocChat initialized successfully")


def _shutdown():
    """Clean up resources."""
    if _llm_router:
        _llm_router.close()
    if _vector_db:
        _vector_db.close()
    if _memory:
        _memory.close()


# ─── Pydantic Models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    sources_used: list
    retrieval_count: int

class URLRequest(BaseModel):
    urls: List[str]

class YouTubeRequest(BaseModel):
    url: str

# ─── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _initialize()
    yield
    _shutdown()


app = FastAPI(
    title="DocChat",
    description="Upload documents, ask questions, get cited answers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok" if _llm_router else "not_initialized",
        "llm": _llm_router.health_check() if _llm_router else {},
    }


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process document files (PDF, TXT, MD)."""
    if not _doc_processor:
        raise HTTPException(status_code=503, detail="Not initialized")

    results = []
    for uploaded_file in files:
        try:
            suffix = Path(uploaded_file.filename or "").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await uploaded_file.read()
                tmp.write(content)
                temp_path = tmp.name

            chunks = _doc_processor.process_document(temp_path)

            for chunk in chunks:
                chunk.source_file = uploaded_file.filename

            if chunks:
                embedded = _embedding_generator.generate_embeddings(chunks)
                _vector_db.insert_embeddings(embedded)

                _memory.save_source({
                    "name": uploaded_file.filename,
                    "type": "Document",
                    "size": f"{len(content) / 1024:.1f} KB",
                    "chunks": len(chunks),
                })

                results.append({
                    "file": uploaded_file.filename,
                    "chunks": len(chunks),
                    "status": "ok",
                })

            os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Upload error for {uploaded_file.filename}: {e}")
            results.append({"file": uploaded_file.filename, "error": str(e)})

    return {"results": results}


@app.post("/api/url")
async def add_urls(request: URLRequest):
    """Scrape web URLs and add as sources."""
    if not _web_scraper:
        raise HTTPException(status_code=503, detail="Not initialized")

    results = []
    for url in request.urls:
        try:
            chunks = _web_scraper.scrape_url(url)
            if chunks:
                embedded = _embedding_generator.generate_embeddings(chunks)
                _vector_db.insert_embeddings(embedded)

                source_name = chunks[0].source_file  # Title extracted by scraper
                _memory.save_source({
                    "name": source_name,
                    "type": "Website",
                    "size": f"{len(chunks)} chunks",
                    "chunks": len(chunks),
                })

                results.append({"url": url, "title": source_name, "chunks": len(chunks), "status": "ok"})
            else:
                results.append({"url": url, "error": "No content extracted"})
        except Exception as e:
            logger.error(f"URL scrape error for {url}: {e}")
            results.append({"url": url, "error": str(e)})

    return {"results": results}


@app.post("/api/youtube")
async def add_youtube(request: YouTubeRequest):
    """Extract YouTube transcript and add as source."""
    if not _youtube_extractor:
        raise HTTPException(status_code=503, detail="Not initialized")

    try:
        chunks = _youtube_extractor.extract_transcript(request.url)
        if chunks:
            embedded = _embedding_generator.generate_embeddings(chunks)
            _vector_db.insert_embeddings(embedded)

            source_name = chunks[0].source_file  # Video title
            _memory.save_source({
                "name": source_name,
                "type": "YouTube",
                "size": f"{len(chunks)} chunks",
                "chunks": len(chunks),
            })

            return {"title": source_name, "chunks": len(chunks), "status": "ok"}
        else:
            raise HTTPException(status_code=400, detail="No transcript found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"YouTube error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about your sources."""
    if not _rag_generator:
        raise HTTPException(status_code=503, detail="Not initialized")

    conv_context = _memory.get_conversation_context(max_turns=5)

    result = _rag_generator.generate_response(
        query=request.query,
        conversation_context=conv_context,
    )

    _memory.save_conversation_turn(result)

    return ChatResponse(
        response=result.response,
        sources_used=result.sources_used,
        retrieval_count=result.retrieval_count,
    )


@app.get("/api/sources")
async def get_sources():
    """List all added sources."""
    if not _memory:
        return {"sources": []}
    return {"sources": _memory.get_sources()}


@app.delete("/api/sources/{source_id}")
async def delete_source(source_id: int):
    """Remove a source by ID."""
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    deleted = _memory.delete_source(source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"status": "ok"}


@app.get("/api/history")
async def get_history():
    """Get chat history."""
    if not _memory:
        return {"history": []}
    return {"history": _memory.get_chat_history()}


# ─── Static Frontend ───────────────────────────────────────────────────────────

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def serve_frontend():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "DocChat API is running. Frontend not found at /static/index.html"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
