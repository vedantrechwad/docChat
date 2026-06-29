# CarnetLM

**Upload research sources. Organize by subject. Ask questions. Write notes with AI assistance.**

CarnetLM is a private, local-first AI-powered document research assistant and Q&A workspace. It allows you to organize your literature, articles, web pages, and video transcripts into isolated subject notebooks, search and chat with them using semantic-hybrid RAG with page citations, and study using an interactive, Google Keep-style spaced repetition deck.

---

## Key Features

- 🔒 **100% Local Privacy**: Your uploaded PDF files, vector databases, SQLite workspace metadata, and text memories are saved locally in the git-ignored `data/` directory. No telemetry, tracking, or cloud storage. A fresh clone gives you a completely clean environment.
- 📂 **Multi-Notebook Vaults**: Separate your projects, courses, or research themes into isolated workspaces. Support for **Private Notebooks** protected with custom passwords (client-side credential verification) to lock sensitive files.
- 📁 **Multi-Format Source Ingestion**:
  - **Documents** — Local PDF, TXT, and Markdown files.
  - **OCR Support** — Extract text automatically from uploaded images.
  - **Web Scraping** — Clean article extraction from any website URL.
  - **YouTube Transcripts** — Extract text subtitles directly from YouTube links.
  - **Clipboard** — Paste raw textual ideas directly as references.
- 💬 **Grounded RAG Q&A with Citations**: Ask natural language questions and receive streaming answers. Every response is grounded in your sources with precise citations mapping back to specific paragraphs and pages.
- 🧠 **Smart Flashcard Study Decks**:
  - **Auto-Generate Concepts** — Generate flashcard decks from your notebooks using AI.
  - **HTML5 Drag-and-Drop** — Order cards in a flexible Google Keep-style grid layout.
  - **Leitner Spaced Repetition** — Grade card reviews to progress them through Leitner study boxes (Box 1 to 5).
  - **Deck Insights Sidebar** — Monitor your active studying progress bars and reset memory counts on demand.
- ✍️ **Integrated Editor & AI Assist**: Compose synthesis documents in an editor workspace. Use AI to correct grammar, simplify complex sentences, expand paragraphs, write definitions, or rewrite text.
- 🔊 **Local Text-to-Speech (TTS)**: Health-checked integration with Orpheus TTS to read syntheses and chat answers out loud locally.
- ⚡ **One-Click Startup**: Windows launcher batch file (`run.bat`) automatically configures a python virtual environment, installs dependencies safely, and launches the web app.

---

## Tech Stack

### Frontend (Single Page Application)
- **Structure**: Vanilla HTML5 + SVG icons (`icons.js`).
- **Styling**: Highly polished, modern dark-themed responsive CSS (`theme.css`) utilizing grid layouts and transitions.
- **Interactivity**: HTML5 Drag-and-Drop API, canvas previews, and client-side password hashing.
- **State Management**: Zero-dependencies JavaScript client synced with `localStorage` for tab and active notebook states.

### Backend (FastAPI API Server)
- **Web Server**: FastAPI framework powered by Uvicorn.
- **Vector Search**: Milvus Lite (embedded file-based database running from `data/docchat.db` — no standalone server installation needed).
- **Embeddings Generator**: Local FastEmbed library (running `BAAI/bge-small-en-v1.5` transformer model).
- **Hybrid Search**: BM25 keyword matching via the `bm25s` library.
- **Data Storage**: SQLite engine (`data/memory.db`) for tracking notebook settings, study decks, notes, and password credentials.
- **Document Processing**: `pymupdf` (PDF content extraction) and `easyocr` (OCR image scanning).
- **Web Scraping**: `httpx`, `beautifulsoup4`, and `trafilatura` for clean content extraction.
- **YouTube Extractors**: `yt-dlp` for video subtitles parsing.
- **LLM Engine**: Google Gemini 2.5 Flash API client (default) with offline local Ollama model backup fallbacks.

---

## Directory Structure

```
CarnetLM/
├── backend/            # FastAPI main router, lifespan managers, and models
│   ├── helpers.py      # Prompt templates, export formats, synthesis functions
│   └── main.py         # Main webapp route definitions & processing logic
├── src/                # Modular Python sub-packages
│   ├── discovery/      # Web search tools (DuckDuckGo integration)
│   ├── document_processing/ # PDF text extraction and OCR image support
│   ├── embeddings/     # Local FastEmbed transformer wrapper
│   ├── generation/     # RAG prompt generation, routing and retrieval
│   ├── ingest/         # Content chunking and background ingest pipelines
│   ├── llm/            # LLMRouter supporting Gemini API & local Ollama fallback
│   ├── memory/         # Local SQLite storage (notes, flashcards, notebooks metadata)
│   ├── tts/            # Local Orpheus TTS connector client
│   └── vector_database/# Local Milvus Lite wrapper
├── static/             # Responsive frontend Single Page Application
│   ├── index.html      # Main app viewport, modals, study desks and tabs
│   ├── theme.css       # Curator dark palette theme, styles, transitions, grid systems
│   └── icons.js        # Lucide vector icon components
├── run.bat             # Automated Windows setup and startup file
├── pyproject.toml      # Project packaging metadata and dependencies
└── requirements.txt    # Standard package-requirements manifest
```

---

## Quick Start

### Windows Setup (Automatic)

1. Clone this repository to your computer.
2. Create a `.env` file in the root folder (see `.env.example` as a template) and add your `GEMINI_API_KEY`.
3. Double-click the `run.bat` file.
   - It will automatically check for the `uv` tool manager.
   - It runs dependency synchronization (`uv sync`).
   - It launches the browser and opens **http://localhost:8000** automatically.

### macOS / Linux Setup (Manual)

1. Ensure you have **Python 3.11+** installed.
2. Install the [uv package manager](https://docs.astral.sh/uv/) (highly recommended):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Set up your environment file:
   ```bash
   cp .env.example .env
   # Open .env and insert your GEMINI_API_KEY
   ```
4. Install all dependencies:
   ```bash
   uv sync
   ```
5. Run the web server:
   ```bash
   uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
6. Open your web browser to **http://localhost:8000**.

### Optional: Ollama Offline Fallback

For fully offline/local LLM execution:
1. Install [Ollama](https://ollama.ai).
2. Pull your model of choice (e.g. `llama3` or `phi3`):
   ```bash
   ollama pull llama3
   ```
3. Configure your `.env` file:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3
   ```

---

## API Endpoints Directory

Below is the directory of routes defined inside `backend/main.py`:

| Method | Endpoint | Description |
|:---|:---|:---|
| **Health & Configuration** | | |
| `GET` | `/api/health` | Diagnostic state of LLM models, Milvus, SQLite db, and TTS engine |
| `GET` | `/api/models` | List all available local Ollama and cloud Gemini models |
| `POST` | `/api/models/select` | Set the active model selection for the session |
| `GET` | `/api/chunking/profiles` | Fetch chunking profiles (paragraphs, words, tokens) |
| `GET` | `/api/settings/chunking` | Retrieve current notebook chunking setup parameters |
| `PUT` | `/api/settings/chunking` | Save/modify chunk tokens and overlap tolerances |
| `GET` | `/api/settings/performance` | Query performance settings (Quality vs. Fast speed profiles) |
| `PUT` | `/api/settings/performance` | Save performance settings profile |
| `GET` | `/api/settings/discover` | Query web discovery preferences |
| `PUT` | `/api/settings/discover` | Modify web discovery settings |
| **Web Discovery Search** | | |
| `POST` | `/api/discover/search` | Search online using DuckDuckGo search providers |
| `POST` | `/api/discover/ingest` | Download and ingest search result target pages |
| **Ingestion Pipeline** | | |
| `GET` | `/api/ingest/jobs` | Get active document-parsing job queue lists |
| `GET` | `/api/ingest/jobs/{job_id}` | Check status details on an ongoing file ingestion job |
| `POST` | `/api/upload` | Upload PDF, TXT, MD documents (supports OCR image extract) |
| `POST` | `/api/url` | Extract and ingest content from website articles |
| `POST` | `/api/youtube` | Crawl transcripts from YouTube videos |
| `GET` | `/api/sources` | Get list of all source references in the current notebook |
| `DELETE`| `/api/sources/{source_id}` | Remove a source reference and clear its vector store chunks |
| `GET` | `/api/sources/{source_id}/content` | Retrieve raw text contents of an ingested document |
| `PUT` | `/api/sources/{source_id}/content` | Modify or write edits back to an ingested source |
| `POST` | `/api/sources/refresh` | Re-run ingestion crawl pipelines on a document |
| **Notebook Workspaces** | | |
| `GET` | `/api/notebooks` | Fetch list of active subject notebooks |
| `POST` | `/api/notebooks` | Create a new subject notebook (supports password protection) |
| `POST` | `/api/notebooks/{notebook_id}/verify`| Authenticate credentials for password-locked notebooks |
| `PUT` | `/api/notebooks/{notebook_id}` | Rename notebook workspaces |
| `DELETE`| `/api/notebooks/{notebook_id}` | Drop a notebook and remove all associated local sources and db indices |
| **Workspace Chat & Search** | | |
| `POST` | `/api/chat/stream` | Stream chatbot query answers using semantic-hybrid RAG |
| `GET` | `/api/history` | Retrieve full chat history for the active notebook |
| `DELETE`| `/api/history` | Wipe chat log history clean |
| `POST` | `/api/search` | Execute quick semantic queries over local vector databases |
| `POST` | `/api/chat/export` | Export chat logs as Markdown or Text transcripts |
| **Synthesis & Research** | | |
| `GET` | `/api/summary` | Query active synthesis summaries generated from local source contexts |
| `POST` | `/api/summary` | Force-generate a brand-new notebook synthesis summary |
| `POST` | `/api/compare` | Synthesize side-by-side comparisons of multiple sources |
| `POST` | `/api/clipboard` | Quick-ingest pasted clipboard texts as source references |
| **Notes & Document Workspace**| | |
| `GET` | `/api/notebooks/{notebook_id}/notes` | Fetch notes written inside the active notebook workspace |
| `POST` | `/api/notebooks/{notebook_id}/notes` | Create a new note |
| `GET` | `/api/notes/{note_id}` | Retrieve individual note details |
| `PUT` | `/api/notes/{note_id}` | Update note title and details |
| `POST` | `/api/notes/{note_id}/append`| Append text strings directly to a note |
| `DELETE`| `/api/notes/{note_id}` | Delete notes from SQLite databases |
| `POST` | `/api/notes/{note_id}/index` | Toggle indexing note content into the RAG vector search database |
| `GET` | `/api/notebooks/{notebook_id}/document` | Get compiled synthesis document content |
| `PUT` | `/api/notebooks/{notebook_id}/document` | Write compiler edits back to active synthesis documents |
| **AI Assist & Document Export**| | |
| `POST` | `/api/ai/assist` | Run AI editor operations (simplify, define, expand, grammar, rewrite) |
| `POST` | `/api/export` | Export notebooks to DOCX, PDF, TXT, or Markdown formats |
| **Flashcard Concepts (Leitner System)** | | |
| `POST` | `/api/concepts/generate`| Trigger AI generation of study flashcards from ingested sources |
| `GET` | `/api/concepts` | Fetch flashcards deck list for the active notebook workspace |
| `POST` | `/api/concepts` | Manually insert a newly designed card (prepended to index 0) |
| `POST` | `/api/concepts/reorder` | Persist customized Keep-style drag-and-drop index sorting |
| `POST` | `/api/concepts/grade` | Grade card reviews to progress boxes (Leitner levels 1 to 5) |
| `DELETE`| `/api/concepts/{concept_id}` | Remove a concept card from study decks |
| `POST` | `/api/notebooks/{notebook_id}/concepts/reset-progress`| Wipe spaced repetition history, resetting all cards to Box 1 |
| **Text-to-Speech** | | |
| `GET` | `/api/tts/health` | Check Local Orpheus TTS health connection status |
| `POST` | `/api/tts` | Convert text to speech audio waveforms |
| **Static Viewport Router** | | |
| `GET` | `/` | Serves the responsive SPA web frontend |

---

## Data Privacy Promise

CarnetLM keeps all uploaded literature and generated memories secure. The databases (`memory.db` & `docchat.db`), source raw files, and RAG embeddings cache live strictly inside your local repository folder `data/` which is configured in `.gitignore`. Your data is never synced to GitHub, and pulling the latest updates from git maintains your completely clean local space.

---

## License

This project is licensed under the [MIT License](LICENSE).