"""
Study Companion FastAPI Backend

Main application entry point. Provides REST API endpoints for:
- Document upload & processing
- RAG-based chat with citations
- Quiz & flashcard generation
- Study planning
- Concept maps
- Canvas AI operations
- Model settings management
"""

import os
import uuid
import time
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Pydantic Models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_chunks: int = 8

class ChatResponse(BaseModel):
    response: str
    sources_used: List[Dict[str, Any]]
    retrieval_count: int

class QuizRequest(BaseModel):
    source_name: Optional[str] = None
    num_questions: int = 10
    question_types: List[str] = ["mcq", "true_false", "short_answer"]
    difficulty: str = "mixed"

class FlashcardRequest(BaseModel):
    source_name: Optional[str] = None
    num_cards: int = 20
    style: str = "concept"
    topic: Optional[str] = None

class AudioGuideRequest(BaseModel):
    source_name: Optional[str] = None
    guide_type: str = "lecture"
    target_duration: str = "5 minutes"
    topic: Optional[str] = None

class StudyPlanRequest(BaseModel):
    source_name: Optional[str] = None
    total_days: int = 14
    hours_per_day: float = 3.0
    exam_date: Optional[str] = None

class ConceptMapRequest(BaseModel):
    source_name: Optional[str] = None
    max_concepts: int = 15

class CanvasAIRequest(BaseModel):
    text: str
    action: str  # "explain", "summarize", "expand", "simplify", "grammar", "translate", "mnemonic"
    context: str = ""

class URLRequest(BaseModel):
    urls: List[str]

class TextRequest(BaseModel):
    text: str

class YouTubeRequest(BaseModel):
    url: str

class ModelSettingsRequest(BaseModel):
    model_mode: str = "local"
    cloud_provider: str = "gemini"
    primary_model: str = "llama3"
    fast_model: str = "phi3"

class HealthResponse(BaseModel):
    status: str
    models: Dict[str, Any]
    sources_count: int

class ELI5Request(BaseModel):
    term: str
    context: str = ""

class GlossaryRequest(BaseModel):
    content: str
    max_terms: int = 20

class FocusCheckRequest(BaseModel):
    content: str

class GradeAnswerRequest(BaseModel):
    question: Dict[str, Any]
    user_answer: str


# ─── Application State ────────────────────────────────────────────────────────

class AppState:
    """Holds all initialized pipeline components."""

    def __init__(self):
        self.initialized = False
        self.session_id = str(uuid.uuid4())
        self.sources: List[Dict[str, Any]] = []

        # Components (initialized lazily)
        self.llm_router = None
        self.doc_processor = None
        self.embedding_generator = None
        self.vector_db = None
        self.rag_generator = None
        self.memory = None
        self.audio_transcriber = None
        self.youtube_transcriber = None
        self.web_scraper = None
        self.quiz_generator = None
        self.flashcard_generator = None
        self.study_planner = None
        self.concept_map_generator = None
        self.audio_guide_generator = None
        self.micro_ai = None
        self.model_mode = "local"
        self.cloud_provider = "gemini"

    def initialize(
        self,
        primary_model: str = "llama3",
        fast_model: str = "phi3",
        model_mode: str = "local",
        cloud_provider: str = "gemini",
    ):
        """Initialize all pipeline components."""
        from src.llm.llm_router import create_default_router
        from src.document_processing.doc_processor import DocumentProcessor
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.vector_database.milvus_vector_db import MilvusVectorDB
        from src.generation.rag_v2 import RAGGeneratorV2
        from src.memory.local_memory import LocalMemoryLayer
        from src.study_tools import (
            QuizGenerator, FlashcardGenerator, StudyPlanner,
            ConceptMapGenerator, MicroAITools,
        )
        from src.audio.audio_study_guide import AudioStudyGuideGenerator
        logger.info("Initializing Study Companion pipeline...")
        self.model_mode = model_mode if model_mode in {"local", "cloud"} else "local"
        self.cloud_provider = cloud_provider if cloud_provider in {"groq", "openai", "gemini"} else "gemini"

        # 1. LLM Router
        self.llm_router = create_default_router(
            ollama_primary=primary_model,
            ollama_fast=fast_model,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            prefer_cloud=self.model_mode == "cloud",
            cloud_provider=self.cloud_provider,
        )

        # 2. Document Processing
        self.doc_processor = DocumentProcessor()

        # 3. Embeddings (keep fastembed for now, works without Ollama)
        self.embedding_generator = EmbeddingGenerator()

        # 4. Vector DB
        self.vector_db = MilvusVectorDB(
            db_path=f"./data/milvus_{self.session_id[:8]}.db",
            collection_name=f"collection_{self.session_id[:8]}",
        )

        # 5. RAG Generator
        self.rag_generator = RAGGeneratorV2(
            llm_router=self.llm_router,
            embedding_generator=self.embedding_generator,
            vector_db=self.vector_db,
        )

        # 6. Local Memory
        self.memory = LocalMemoryLayer(
            user_id="study_user",
            session_id=self.session_id,
            create_new_session=True,
        )

        # 7. Audio (lazy init — only loads model when first used)
        try:
            from src.audio_processing.local_transcriber import LocalAudioTranscriber, LocalYouTubeTranscriber
            self.audio_transcriber = LocalAudioTranscriber(model_size="base")
            self.youtube_transcriber = LocalYouTubeTranscriber(model_size="base")
        except Exception as e:
            logger.warning(f"Audio transcription not available: {e}")

        # 8. Web Scraper
        try:
            from src.web_scraping.local_scraper import LocalWebScraper
            self.web_scraper = LocalWebScraper()
        except Exception as e:
            logger.warning(f"Web scraping not available: {e}")

        # 9. Study Tools
        self.quiz_generator = QuizGenerator(self.llm_router)
        self.flashcard_generator = FlashcardGenerator(self.llm_router)
        self.study_planner = StudyPlanner(self.llm_router)
        self.concept_map_generator = ConceptMapGenerator(self.llm_router)
        self.micro_ai = MicroAITools(self.llm_router)
        self.audio_guide_generator = AudioStudyGuideGenerator(self.llm_router)

        self.initialized = True
        logger.info("✅ Study Companion pipeline initialized successfully!")

    def get_content_for_source(self, source_name: Optional[str] = None) -> str:
        """Retrieve stored content for a source (or all sources)."""
        if not self.vector_db:
            return ""

        query = f"content from {source_name}" if source_name else "main topics overview"
        query_vector = self.embedding_generator.generate_query_embedding(query)

        filter_expr = f'source_file == "{source_name}"' if source_name else None
        results = self.vector_db.search(
            query_vector=query_vector.tolist(),
            limit=30,
            filter_expr=filter_expr,
        )

        if results:
            results.sort(key=lambda x: x.get("chunk_index", 0))
            return "\n\n".join(r["content"] for r in results)
        return ""


# Global state
state = AppState()


# ─── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the pipeline on startup."""
    try:
        state.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize pipeline on startup: {e}")
        logger.info("Pipeline will be initialized on first request or via /api/settings")
    yield
    # Cleanup
    if state.memory:
        state.memory.close()
    if state.llm_router:
        state.llm_router.close()


app = FastAPI(
    title="Study Companion API",
    description="AI-powered study companion with RAG, quizzes, flashcards, and more",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health & Settings ─────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check system health and model availability."""
    models_status = {}
    if state.llm_router:
        models_status = state.llm_router.health_check()
    models_status["selection"] = {
        "mode": state.model_mode,
        "cloud_provider": state.cloud_provider,
    }
    return HealthResponse(
        status="ok" if state.initialized else "not_initialized",
        models=models_status,
        sources_count=len(state.sources),
    )


@app.post("/api/settings")
async def update_settings(settings: ModelSettingsRequest):
    """Update model settings and reinitialize the pipeline."""
    try:
        state.initialize(
            primary_model=settings.primary_model,
            fast_model=settings.fast_model,
            model_mode=settings.model_mode,
            cloud_provider=settings.cloud_provider,
        )
        return {"status": "ok", "message": "Settings updated and pipeline reinitialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def list_models():
    """List available Ollama models."""
    if not state.llm_router:
        return {"models": []}
    models = state.llm_router.list_available_ollama_models()
    return {"models": models}


# ─── Sources ───────────────────────────────────────────────────────────────────

@app.get("/api/sources")
async def get_sources():
    """Get all uploaded sources."""
    return {"sources": state.sources}


@app.post("/api/sources/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process document files."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    results: List[Dict[str, Any]] = []
    for uploaded_file in files:
        try:
            suffix = Path(uploaded_file.filename or "").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await uploaded_file.read()
                tmp.write(content)
                temp_path = tmp.name

            if suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
                if state.audio_transcriber:
                    chunks = state.audio_transcriber.transcribe_audio(temp_path)
                    source_type = "Audio"
                else:
                    results.append({"file": uploaded_file.filename, "error": "Audio transcription not available"})
                    os.unlink(temp_path)
                    continue
            else:
                chunks = state.doc_processor.process_document(temp_path)
                source_type = "Document"

            for chunk in chunks:
                chunk.source_file = uploaded_file.filename

            if chunks:
                embedded = state.embedding_generator.generate_embeddings(chunks)
                if len(state.sources) == 0:
                    state.vector_db.create_index(use_binary_quantization=False)
                state.vector_db.insert_embeddings(embedded)

                source_info = {
                    "name": uploaded_file.filename,
                    "type": source_type,
                    "size": f"{len(content) / 1024:.1f} KB",
                    "chunks": len(chunks),
                    "uploaded_at": time.strftime("%Y-%m-%d %H:%M"),
                }
                state.sources.append(source_info)
                results.append({"file": uploaded_file.filename, "chunks": len(chunks), "status": "ok"})

            os.unlink(temp_path)

        except Exception as e:
            results.append({"file": uploaded_file.filename, "error": str(e)})

    return {"results": results, "total_sources": len(state.sources)}


@app.post("/api/sources/url")
async def process_urls(request: URLRequest):
    """Scrape and process web URLs."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not state.web_scraper:
        raise HTTPException(status_code=503, detail="Web scraper not available")

    results = []
    for url in request.urls:
        try:
            chunks = state.web_scraper.scrape_url(url)
            if chunks:
                for chunk in chunks:
                    chunk.source_file = url
                embedded = state.embedding_generator.generate_embeddings(chunks)
                if len(state.sources) == 0:
                    state.vector_db.create_index(use_binary_quantization=False)
                state.vector_db.insert_embeddings(embedded)

                source_info = {
                    "name": url,
                    "type": "Website",
                    "size": f"{len(chunks)} chunks",
                    "chunks": len(chunks),
                    "uploaded_at": time.strftime("%Y-%m-%d %H:%M"),
                }
                state.sources.append(source_info)
                results.append({"url": url, "chunks": len(chunks), "status": "ok"})
            else:
                results.append({"url": url, "error": "No content extracted"})
        except Exception as e:
            results.append({"url": url, "error": str(e)})

    return {"results": results}


@app.post("/api/sources/youtube")
async def process_youtube(request: YouTubeRequest):
    """Process a YouTube video."""
    if not state.youtube_transcriber:
        raise HTTPException(status_code=503, detail="YouTube transcription not available")

    try:
        chunks = state.youtube_transcriber.transcribe_youtube_video(request.url)
        if chunks:
            embedded = state.embedding_generator.generate_embeddings(chunks)
            if len(state.sources) == 0:
                state.vector_db.create_index(use_binary_quantization=False)
            state.vector_db.insert_embeddings(embedded)

            video_id = state.youtube_transcriber.extract_video_id(request.url)
            source_info = {
                "name": f"YouTube Video {video_id}",
                "type": "YouTube Video",
                "size": f"{len(chunks)} segments",
                "chunks": len(chunks),
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M"),
                "url": request.url,
            }
            state.sources.append(source_info)
            return {"status": "ok", "chunks": len(chunks), "video_id": video_id}

        return {"status": "error", "message": "No transcript content extracted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sources/text")
async def process_text(request: TextRequest):
    """Process pasted text content."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write(request.text)
            temp_path = tmp.name

        chunks = state.doc_processor.process_document(temp_path)
        name = f"Pasted Text ({time.strftime('%H:%M')})"
        for chunk in chunks:
            chunk.source_file = name

        if chunks:
            embedded = state.embedding_generator.generate_embeddings(chunks)
            if len(state.sources) == 0:
                state.vector_db.create_index(use_binary_quantization=False)
            state.vector_db.insert_embeddings(embedded)

            state.sources.append({
                "name": name,
                "type": "Text",
                "size": f"{len(request.text)} chars",
                "chunks": len(chunks),
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M"),
            })

        os.unlink(temp_path)
        return {"status": "ok", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG-powered chat with citations."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    conv_context = ""
    if state.memory:
        conv_context = state.memory.get_conversation_context(max_turns=5)

    result = state.rag_generator.generate_response(
        query=request.query,
        max_chunks=request.max_chunks,
        conversation_context=conv_context,
    )

    if state.memory:
        state.memory.save_conversation_turn(result)

    return ChatResponse(
        response=result.response,
        sources_used=result.sources_used,
        retrieval_count=result.retrieval_count,
    )


# ─── Study Tools ───────────────────────────────────────────────────────────────

@app.post("/api/quiz/generate")
async def generate_quiz(request: QuizRequest):
    """Generate a quiz from uploaded sources."""
    if not state.quiz_generator:
        raise HTTPException(status_code=503, detail="Quiz generator not available")

    content = state.get_content_for_source(request.source_name)
    if not content:
        raise HTTPException(status_code=404, detail="No content found for quiz generation")

    quiz = state.quiz_generator.generate_quiz(
        content=content,
        source_name=request.source_name or "All Sources",
        num_questions=request.num_questions,
        question_types=request.question_types,
        difficulty=request.difficulty,
    )
    return quiz.to_dict()


@app.post("/api/quiz/grade")
async def grade_answer(request: GradeAnswerRequest):
    """Grade a quiz answer."""
    if not state.quiz_generator:
        raise HTTPException(status_code=503, detail="Quiz generator not available")

    from src.study_tools import QuizQuestion
    question = QuizQuestion(
        question=request.question.get("question", ""),
        question_type=request.question.get("type", "mcq"),
        options=request.question.get("options", []),
        correct_answer=request.question.get("correct_answer", ""),
        explanation=request.question.get("explanation", ""),
    )
    result = state.quiz_generator.grade_answer(question, request.user_answer)
    return result


@app.post("/api/flashcards/generate")
async def generate_flashcards(request: FlashcardRequest):
    """Generate flashcards from uploaded sources."""
    if not state.flashcard_generator:
        raise HTTPException(status_code=503, detail="Flashcard generator not available")

    content = state.get_content_for_source(request.source_name)
    if not content:
        raise HTTPException(status_code=404, detail="No content found")

    deck = state.flashcard_generator.generate_flashcards(
        content=content,
        source_name=request.source_name or "All Sources",
        num_cards=request.num_cards,
        style=request.style,
    )
    return deck.to_dict()


@app.post("/api/study-plan/generate")
async def generate_study_plan(request: StudyPlanRequest):
    """Generate a study plan from uploaded sources."""
    if not state.study_planner:
        raise HTTPException(status_code=503, detail="Study planner not available")

    content = state.get_content_for_source(request.source_name)
    if not content:
        raise HTTPException(status_code=404, detail="No content found")

    plan = state.study_planner.generate_study_plan(
        content=content,
        source_name=request.source_name or "All Sources",
        total_days=request.total_days,
        hours_per_day=request.hours_per_day,
        exam_date=request.exam_date,
    )
    return {
        "title": plan.title,
        "total_days": plan.total_days,
        "schedule": [
            {
                "day": d.day,
                "date": d.date,
                "topics": d.topics,
                "activities": d.activities,
                "estimated_hours": d.estimated_hours,
                "notes": d.notes,
            }
            for d in plan.schedule
        ],
    }


@app.post("/api/concept-map/generate")
async def generate_concept_map(request: ConceptMapRequest):
    """Generate a concept map from uploaded sources."""
    if not state.concept_map_generator:
        raise HTTPException(status_code=503, detail="Concept map generator not available")

    content = state.get_content_for_source(request.source_name)
    if not content:
        raise HTTPException(status_code=404, detail="No content found")

    result = state.concept_map_generator.generate_concept_map(
        content=content,
        source_name=request.source_name or "All Sources",
        max_concepts=request.max_concepts,
    )
    return result


@app.post("/api/audio/generate")
async def generate_audio_guide(request: AudioGuideRequest):
    """Generate an audio study guide script from uploaded sources."""
    if not state.audio_guide_generator:
        raise HTTPException(status_code=503, detail="Audio generator not available")

    content = state.get_content_for_source(request.topic) # we can use topic as source name filter or fallback
    if not content:
        raise HTTPException(status_code=404, detail="No content found")

    import json
    guide = state.audio_guide_generator.generate_from_text(
        text_content=content,
        source_name=request.topic or "All Sources",
        guide_type=request.guide_type,
        target_duration=request.target_duration,
    )
    # The frontend expects the JSON representation or dict
    return json.loads(guide.to_json())


# ─── Canvas AI ─────────────────────────────────────────────────────────────────

@app.post("/api/canvas/ai")
async def canvas_ai_action(request: CanvasAIRequest):
    """AI actions for the canvas (explain, summarize, expand, etc.)."""
    if not state.micro_ai:
        raise HTTPException(status_code=503, detail="AI tools not available")

    action_map = {
        "explain": lambda: state.micro_ai.explain_like_im_5(request.text, request.context),
        "simplify": lambda: state.micro_ai.simplify_text(request.text),
        "expand": lambda: state.micro_ai.expand_text(request.text, request.context),
        "grammar": lambda: state.micro_ai.fix_grammar(request.text),
    }

    if request.action == "summarize":
        from src.llm.llm_router import TaskType
        response = state.llm_router.generate(
            prompt=f"Summarize the following text concisely:\n\n{request.text}",
            task_type=TaskType.SUMMARIZE,
            temperature=0.2,
            max_tokens=500,
        )
        return {"result": response.content}

    if request.action == "translate":
        from src.llm.llm_router import TaskType
        response = state.llm_router.generate(
            prompt=f"Translate the following text to English. If it's already in English, translate to Spanish:\n\n{request.text}",
            task_type=TaskType.TRANSLATE,
            temperature=0.1,
            max_tokens=500,
        )
        return {"result": response.content}

    if request.action == "mnemonic":
        items = [item.strip() for item in request.text.split(",")]
        result = state.micro_ai.generate_mnemonic(items)
        return {"result": result}

    if request.action in action_map:
        result = action_map[request.action]()
        return {"result": result}

    raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")


# ─── Micro AI Endpoints ───────────────────────────────────────────────────────

@app.post("/api/ai/eli5")
async def eli5(request: ELI5Request):
    """Explain Like I'm 5."""
    if not state.micro_ai:
        raise HTTPException(status_code=503, detail="AI tools not available")
    result = state.micro_ai.explain_like_im_5(request.term, request.context)
    return {"explanation": result}


@app.post("/api/ai/glossary")
async def extract_glossary(request: GlossaryRequest):
    """Extract glossary from content."""
    if not state.micro_ai:
        raise HTTPException(status_code=503, detail="AI tools not available")
    glossary = state.micro_ai.extract_glossary(request.content, request.max_terms)
    return {"glossary": glossary}


@app.post("/api/ai/focus-check")
async def focus_check(request: FocusCheckRequest):
    """Generate a focus check question."""
    if not state.micro_ai:
        raise HTTPException(status_code=503, detail="AI tools not available")
    question = state.micro_ai.generate_focus_question(request.content)
    return question


@app.get("/api/summary")
async def get_summary(length: str = "medium"):
    """Generate a document summary."""
    if not state.rag_generator:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    result = state.rag_generator.generate_summary(summary_length=length)
    return {
        "summary": result.response,
        "sources_used": result.sources_used,
    }


# ─── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
