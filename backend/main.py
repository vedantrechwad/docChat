"""
DocChat — FastAPI Backend

Multi-notebook document Q&A with notes, AI writing assist, and export.
"""

import os
import re
import time
import logging
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

    Path("./data").mkdir(exist_ok=True)

    _llm_router = LLMRouter(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
        auto_start=True,
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

    try:
        _vector_db.create_index(use_binary_quantization=False)
    except Exception:
        pass

    logger.info("DocChat initialized successfully")


def _shutdown():
    if _llm_router:
        _llm_router.close()
    if _vector_db:
        _vector_db.close()
    if _memory:
        _memory.close()


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

class NotebookRename(BaseModel):
    name: str

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


# ─── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _initialize()
    yield
    _shutdown()

app = FastAPI(
    title="DocChat",
    description="Multi-notebook document Q&A with AI writing assist.",
    version="2.0.0",
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
    return {
        "status": "ok" if _llm_router else "not_initialized",
        "llm": _llm_router.health_check() if _llm_router else {},
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
    }

@app.post("/api/models/select")
async def select_model(request: ModelSelect):
    """Switch the active Ollama model."""
    if not _llm_router:
        raise HTTPException(status_code=503, detail="Not initialized")
    if _llm_router.set_model(request.model):
        return {"status": "ok", "model": request.model}
    raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")


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
    nb_id = _memory.create_notebook(request.name)
    return {"id": nb_id, "name": request.name, "status": "ok"}

@app.put("/api/notebooks/{notebook_id}")
async def rename_notebook(notebook_id: int, request: NotebookRename):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    if _memory.rename_notebook(notebook_id, request.name):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Notebook not found")

@app.delete("/api/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: int):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    if _memory.delete_notebook(notebook_id):
        # Also delete vectors for this notebook from Milvus
        if _vector_db:
            _vector_db.delete_by_notebook(notebook_id)
        return {"status": "ok"}
    raise HTTPException(status_code=400, detail="Cannot delete the last notebook")


# ─── Sources ──────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    notebook_id: int = Form(1),
):
    """Upload and process document files."""
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
                _vector_db.insert_embeddings(embedded, notebook_id=notebook_id)
                _memory.save_source({
                    "name": uploaded_file.filename,
                    "type": "Document",
                    "size": f"{len(content) / 1024:.1f} KB",
                    "chunks": len(chunks),
                }, notebook_id=notebook_id)
                results.append({"file": uploaded_file.filename, "chunks": len(chunks), "status": "ok"})

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
                _vector_db.insert_embeddings(embedded, notebook_id=request.notebook_id)
                source_name = chunks[0].source_file
                _memory.save_source({
                    "name": source_name, "type": "Website",
                    "size": f"{len(chunks)} chunks", "chunks": len(chunks),
                }, notebook_id=request.notebook_id)
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
            _vector_db.insert_embeddings(embedded, notebook_id=request.notebook_id)
            source_name = chunks[0].source_file
            _memory.save_source({
                "name": source_name, "type": "YouTube",
                "size": f"{len(chunks)} chunks", "chunks": len(chunks),
            }, notebook_id=request.notebook_id)
            return {"title": source_name, "chunks": len(chunks), "status": "ok"}
        raise HTTPException(status_code=400, detail="No transcript found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"YouTube error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sources")
async def get_sources(notebook_id: int = Query(1)):
    if not _memory:
        return {"sources": []}
    return {"sources": _memory.get_sources(notebook_id)}


@app.delete("/api/sources/{source_id}")
async def delete_source(source_id: int):
    if not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")
    # Get source info before deleting (needed to clean vectors)
    source_info = _memory.get_source_by_id(source_id)
    if source_info and _memory.delete_source(source_id):
        # Also delete vectors from Milvus
        if _vector_db and source_info:
            _vector_db.delete_by_source(
                source_file=source_info["name"],
                notebook_id=source_info.get("notebook_id"),
            )
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Source not found")


# ─── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about your sources."""
    if not _rag_generator:
        raise HTTPException(status_code=503, detail="Not initialized")

    conv_context = _memory.get_conversation_context(
        notebook_id=request.notebook_id, max_turns=5,
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


# ─── Streaming Chat ────────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream a chat response using Server-Sent Events."""
    import json as _json

    if not _rag_generator:
        raise HTTPException(status_code=503, detail="Not initialized")

    conv_context = _memory.get_conversation_context(
        notebook_id=request.notebook_id, max_turns=5,
    )

    # Use the RAG pipeline for retrieval, but stream the generation
    result = _rag_generator.generate_response(
        query=request.query,
        notebook_id=request.notebook_id,
        conversation_context=conv_context,
    )

    _memory.save_conversation_turn(result, notebook_id=request.notebook_id)

    async def event_generator():
        # Send sources first
        yield f"data: {_json.dumps({'type': 'sources', 'sources_used': result.sources_used, 'retrieval_count': result.retrieval_count})}\n\n"

        # Stream the response in chunks to simulate streaming
        # (True streaming requires refactoring LLM router to yield tokens)
        words = result.response.split(' ')
        chunk_size = 3  # Send 3 words at a time for smooth streaming
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if i + chunk_size < len(words):
                chunk += ' '
            yield f"data: {_json.dumps({'type': 'token', 'content': chunk})}\n\n"

        yield f"data: {_json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Auto-Summary ──────────────────────────────────────────────────────────────

@app.post("/api/summary")
async def generate_summary(request: SummaryRequest):
    """Auto-generate a summary / study guide from all sources."""
    if not _llm_router or not _vector_db or not _embedding_generator:
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

    # Build context from retrieved chunks
    context = "\n\n".join([r["content"] for r in search_results[:15]])
    source_names = list(set(r["citation"]["source_file"] for r in search_results))

    try:
        result = _llm_router.generate(
            prompt=f"""Based on the following source material, create a comprehensive study guide with:
1. **Executive Summary** (2-3 paragraphs)
2. **Key Topics** (bullet points with brief explanations)
3. **Important Details** (notable facts, figures, or concepts)
4. **Connections** (how different topics relate to each other)

Source Material:
{context}

Sources used: {', '.join(source_names)}""",
            system_prompt="You are an expert study guide creator. Create clear, organized, and comprehensive summaries.",
            temperature=0.3,
            max_tokens=2000,
        )
        return {
            "summary": result.content,
            "sources_used": source_names,
            "provider": result.provider,
            "model": result.model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Source Content Viewer ─────────────────────────────────────────────────────

@app.get("/api/sources/{source_id}/content")
async def get_source_content(source_id: int, notebook_id: int = Query(1)):
    """Get the full content of a source (all its chunks)."""
    if not _vector_db or not _memory:
        raise HTTPException(status_code=503, detail="Not initialized")

    source_info = _memory.get_source_by_id(source_id)
    if not source_info:
        raise HTTPException(status_code=404, detail="Source not found")

    # Query all chunks for this source from Milvus
    query_vector = _embedding_generator.generate_query_embedding(source_info["name"])
    results = _vector_db.search(
        query_vector=query_vector.tolist(),
        limit=100,
        notebook_id=notebook_id,
        filter_expr=f'source_file == "{source_info["name"]}"',
    )

    # Sort by chunk_index
    results.sort(key=lambda x: x.get("citation", {}).get("chunk_index", 0))

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
    }


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

    # Get chunks from each source
    source_contents = {}
    for name in request.source_names:
        query_vector = _embedding_generator.generate_query_embedding(name)
        results = _vector_db.search(
            query_vector=query_vector.tolist(),
            limit=5,
            notebook_id=request.notebook_id,
            filter_expr=f'source_file == "{name}"',
        )
        source_contents[name] = "\n".join([r["content"][:500] for r in results])

    # Build comparison prompt
    sources_text = ""
    for name, content in source_contents.items():
        sources_text += f"\n--- Source: {name} ---\n{content}\n"

    try:
        result = _llm_router.generate(
            prompt=f"""Compare and contrast the following sources. Create a structured comparison with:
1. **Common Themes** — What topics/ideas appear in all sources
2. **Key Differences** — Where sources diverge or provide unique information
3. **Comparison Table** — A markdown table comparing key aspects
4. **Synthesis** — What you learn by combining all sources together

{sources_text}""",
            system_prompt="You are an expert analyst. Create clear, structured comparisons. Use markdown formatting.",
            temperature=0.3,
            max_tokens=2000,
        )
        return {
            "comparison": result.content,
            "sources_compared": request.source_names,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Clipboard Paste as Source ────────────────────────────────────────────────

@app.post("/api/clipboard")
async def add_clipboard_source(request: ClipboardRequest):
    """Add pasted text directly as a source."""
    if not _doc_processor or not _embedding_generator or not _vector_db:
        raise HTTPException(status_code=503, detail="Not initialized")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        from src.document_processing.doc_processor import DocumentChunk

        # Create chunks from the pasted text
        chunks = _doc_processor._create_chunks_from_text(
            text=request.text,
            source_file=request.title,
            source_type="clipboard",
        )

        if chunks:
            embedded = _embedding_generator.generate_embeddings(chunks)
            _vector_db.insert_embeddings(embedded, notebook_id=request.notebook_id)
            _memory.save_source({
                "name": request.title,
                "type": "Clipboard",
                "size": f"{len(request.text)} chars",
                "chunks": len(chunks),
            }, notebook_id=request.notebook_id)
            return {"title": request.title, "chunks": len(chunks), "status": "ok"}

        raise HTTPException(status_code=400, detail="No content to process")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

    # Get the URL from source metadata
    # Look up the source metadata from memory
    cursor = _memory.conn.cursor()
    cursor.execute("SELECT metadata_json FROM sources WHERE id = ?", (request.source_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Source metadata not found")

    import json as _json
    metadata = _json.loads(row["metadata_json"] or "{}")
    # The URL might be in the metadata or we can try to find it from the chunks
    source_name = source_info["name"]

    # Delete old vectors
    if _vector_db:
        _vector_db.delete_by_source(source_name, notebook_id=request.notebook_id)

    try:
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
            embedded = _embedding_generator.generate_embeddings(chunks)
            _vector_db.insert_embeddings(embedded, notebook_id=request.notebook_id)
            # Update source metadata
            _memory.delete_source(request.source_id)
            new_source_name = chunks[0].source_file
            _memory.save_source({
                "name": new_source_name,
                "type": source_type,
                "size": f"{len(chunks)} chunks",
                "chunks": len(chunks),
                "url": url,
            }, notebook_id=request.notebook_id)
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
    if _memory.delete_note(note_id):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Note not found")


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
