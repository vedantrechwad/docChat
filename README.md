# CarnetLM

**Upload documents. Ask questions. Get cited answers.**

CarnetLM is a local-first AI-powered document Q&A tool. Upload PDFs, scrape websites, or extract YouTube transcripts — then ask natural language questions and get responses grounded in your sources with proper citations.

## Features

- 📄 **Document Upload** — PDF, TXT, and Markdown files
- 🌐 **Web Scraping** — Extract content from any website
- 🎥 **YouTube Transcripts** — Auto-extract video subtitles
- 💬 **Cited Answers** — Every claim references its source
- 🔍 **Semantic Search** — Find relevant info across all your sources
- 🧠 **Conversation Memory** — Context-aware follow-up questions
- ⚡ **Fast & Local** — Embeddings and vector search run entirely on your machine
- 🎨 **Clean UI** — Premium dark-themed single-page interface

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier)

### Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd CarnetLM

# 2. Create your .env file
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Install dependencies
uv sync

# 4. Run
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Or on Windows, just double-click `run.bat`.

Then open **http://localhost:8000** in your browser.

> **Note**: Runtime data (SQLite database, vector DB, embeddings cache) lives in the `data/` directory and is never shared via git. Each installation maintains its own local data.

### Optional: Ollama Fallback

For fully offline use, you can configure a local Ollama model as fallback:

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3

# Add to your .env:
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

## Architecture

```
Static Frontend (HTML/CSS/JS)  →  FastAPI Backend  →  RAG Pipeline
                                       │
                    ┌──────────────────┤
                    │                  │
              Gemini 2.5 Flash    Local Components
              (or Ollama)         ├── FastEmbed (embeddings)
                                  ├── Milvus Lite (vector DB)
                                  ├── BeautifulSoup (web scraping)
                                  └── yt-dlp (YouTube transcripts)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload document files |
| `POST` | `/api/url` | Scrape website URLs |
| `POST` | `/api/youtube` | Extract YouTube transcript |
| `POST` | `/api/chat` | Ask a question (RAG) |
| `GET` | `/api/sources` | List all sources |
| `DELETE` | `/api/sources/{id}` | Remove a source |
| `GET` | `/api/history` | Get chat history |

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **LLM**: Gemini 2.5 Flash (free tier) with Ollama fallback
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5, runs locally)
- **Vector DB**: Milvus Lite (file-based, no server needed)
- **Document Processing**: PyMuPDF
- **Web Scraping**: httpx + BeautifulSoup
- **YouTube**: yt-dlp subtitle extraction
- **Frontend**: Vanilla HTML/CSS/JS (zero build step)

## License

MIT