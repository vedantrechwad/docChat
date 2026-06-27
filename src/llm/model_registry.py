"""
Model registry — known context windows and chunk recommendations per model.
"""

import re
from typing import Any, Dict, Optional

# Embedding model hard limit (BGE-small-en-v1.5)
EMBEDDING_MAX_TOKENS = 480

CHUNK_PRESETS: Dict[str, Dict[str, int]] = {
    "compact": {"chunk_tokens": 256, "overlap_tokens": 50},
    "balanced": {"chunk_tokens": 384, "overlap_tokens": 100},
    "dense": {"chunk_tokens": 480, "overlap_tokens": 150},
}

# Pattern -> context size (tokens). First match wins.
MODEL_CONTEXT_OVERRIDES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"gemini", re.I), 1_000_000),
    (re.compile(r"llama3.*8b", re.I), 8192),
    (re.compile(r"llama3", re.I), 8192),
    (re.compile(r"gemma.*4b", re.I), 8192),
    (re.compile(r"gemma", re.I), 8192),
    (re.compile(r"mistral", re.I), 8192),
    (re.compile(r"phi", re.I), 4096),
]

# Pattern -> recommended chunk preset for new uploads
MODEL_CHUNK_RECOMMENDATIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"llama3.*8b", re.I), "balanced"),
    (re.compile(r"llama3", re.I), "balanced"),
    (re.compile(r"gemma", re.I), "dense"),
    (re.compile(r"gemini", re.I), "balanced"),
]


def resolve_context_size(model_name: str, detected: Optional[int] = None) -> int:
    """Resolve context window from registry or Ollama-detected value."""
    for pattern, ctx in MODEL_CONTEXT_OVERRIDES:
        if pattern.search(model_name):
            return ctx
    if detected and detected > 0:
        return detected
    return 4096


def recommend_chunk_preset(model_name: str) -> str:
    """Recommend ingest chunk preset for a model (upload-time only)."""
    for pattern, preset in MODEL_CHUNK_RECOMMENDATIONS:
        if pattern.search(model_name):
            return preset
    return "balanced"


def get_chunk_preset(preset_name: str) -> Dict[str, int]:
    """Get token sizes for a named preset."""
    return CHUNK_PRESETS.get(preset_name, CHUNK_PRESETS["balanced"]).copy()


def tokens_to_chars(tokens: int) -> int:
    """Approximate chars from tokens (English ~4 chars/token)."""
    return tokens * 4


def get_active_model_name(router: Any) -> str:
    """Get display name of the active model from LLMRouter."""
    if router.get_active_provider() == "gemini":
        return "gemini-2.5-flash"
    return router.ollama_model
