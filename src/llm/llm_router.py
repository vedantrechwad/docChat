"""
LLM Router — Gemini-first with Ollama auto-start and model selection.

Tries Gemini API first. If not configured or fails, falls back to
a local Ollama model. Can auto-start Ollama and list/switch models.
"""

import os
import sys
import time
import shutil
import logging
import subprocess
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)


class LLMRouter:
    """
    Routes LLM requests to Gemini (primary) or Ollama (fallback).
    Can auto-start Ollama and manage installed models.

    Usage:
        router = LLMRouter()
        response = router.generate("Explain quantum physics")
        models = router.list_models()
        router.set_model("llama3:8b")
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3",
        auto_start: bool = True,
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self._http_client = httpx.Client(timeout=120.0)
        self._ollama_process = None  # Track the subprocess we started
        self._gemini_client = None  # Reusable Gemini client

        # Validate the key isn't a placeholder
        if self.gemini_api_key and "your_" in self.gemini_api_key.lower():
            self.gemini_api_key = None

        self.gemini_available = bool(self.gemini_api_key)

        # Initialize Gemini client once if available
        if self.gemini_available:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
                logger.info("LLM Router: Gemini API configured (primary)")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
                self.gemini_available = False

        # Auto-start Ollama if not running
        if auto_start:
            self._auto_start_ollama()

        self.ollama_available = self._check_ollama()

        # If Ollama is up, auto-select first available model if default isn't installed
        if self.ollama_available:
            models = self.list_models()
            model_names = [m["name"] for m in models]
            if self.ollama_model not in model_names and model_names:
                self.ollama_model = model_names[0]
                logger.info(f"Default model not found, auto-selected: {self.ollama_model}")

        if self.ollama_available:
            logger.info(f"LLM Router: Ollama available, active model: {self.ollama_model}")
        if not self.gemini_available and not self.ollama_available:
            logger.warning("LLM Router: No LLM provider available!")

    # ─── Ollama Management ─────────────────────────────────────────────────

    def _auto_start_ollama(self) -> None:
        """Try to start Ollama if it's not already running."""
        if self._check_ollama():
            logger.info("Ollama is already running")
            return

        # Find the ollama executable
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            # Common Windows install locations
            for path in [
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
                os.path.expandvars(r"%PROGRAMFILES%\Ollama\ollama.exe"),
            ]:
                if os.path.exists(path):
                    ollama_path = path
                    break

        if not ollama_path:
            logger.warning("Ollama not found. Install from https://ollama.ai")
            return

        logger.info(f"Starting Ollama from: {ollama_path}")
        try:
            # Start ollama serve as a background process
            self._ollama_process = subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            # Wait for it to be ready (up to 10 seconds)
            for i in range(20):
                time.sleep(0.5)
                if self._check_ollama():
                    logger.info(f"Ollama started successfully (PID {self._ollama_process.pid})")
                    return

            logger.warning("Ollama process started but not responding")

        except Exception as e:
            logger.warning(f"Failed to start Ollama: {e}")

    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            r = self._http_client.get(f"{self.ollama_base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """List all installed Ollama models."""
        try:
            r = self._http_client.get(f"{self.ollama_base_url}/api/tags", timeout=5.0)
            r.raise_for_status()
            data = r.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                size_bytes = m.get("size", 0)
                size_gb = round(size_bytes / (1024 ** 3), 1) if size_bytes else 0
                models.append({
                    "name": name,
                    "size": f"{size_gb} GB",
                    "size_bytes": size_bytes,
                    "modified_at": m.get("modified_at", ""),
                    "family": m.get("details", {}).get("family", ""),
                    "parameters": m.get("details", {}).get("parameter_size", ""),
                    "quantization": m.get("details", {}).get("quantization_level", ""),
                    "active": name == self.ollama_model,
                })
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def set_model(self, model_name: str) -> bool:
        """Switch the active Ollama model."""
        models = self.list_models()
        model_names = [m["name"] for m in models]
        if model_name not in model_names:
            logger.error(f"Model '{model_name}' not found. Available: {model_names}")
            return False
        self.ollama_model = model_name
        logger.info(f"Active model switched to: {model_name}")
        return True

    # ─── Generation ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Generate a response, trying Gemini first, then Ollama."""

        # Try Gemini first
        if self.gemini_available:
            try:
                return self._generate_gemini(prompt, system_prompt, temperature, max_tokens)
            except Exception as e:
                logger.warning(f"Gemini failed ({e}), trying Ollama fallback...")

        # Fallback to Ollama
        self.ollama_available = self._check_ollama()
        if self.ollama_available:
            try:
                return self._generate_ollama(prompt, system_prompt, temperature, max_tokens)
            except Exception as e:
                logger.error(f"Ollama also failed: {e}")
                raise

        raise ConnectionError(
            "No LLM provider available. Set GEMINI_API_KEY in .env or start Ollama (ollama serve)."
        )

    def _generate_gemini(
        self, prompt: str, system_prompt: Optional[str],
        temperature: float, max_tokens: int,
    ) -> LLMResponse:
        """Generate using the Google GenAI SDK (reuses client)."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        response = self._gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )

        return LLMResponse(
            content=response.text,
            model="gemini-2.5-flash",
            provider="gemini",
        )

    def _generate_ollama(
        self, prompt: str, system_prompt: Optional[str],
        temperature: float, max_tokens: int,
    ) -> LLMResponse:
        """Generate using Ollama's API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,  # Ensure enough context for RAG prompts
            },
            "keep_alive": "5m",  # Keep model loaded between requests
        }

        response = self._http_client.post(
            f"{self.ollama_base_url}/api/chat", json=payload
        )

        # Read actual error from Ollama instead of generic HTTP error
        if response.status_code != 200:
            try:
                err_data = response.json()
                err_msg = err_data.get("error", response.text)
            except Exception:
                err_msg = response.text
            logger.error(f"Ollama error ({response.status_code}): {err_msg}")
            raise RuntimeError(f"Ollama error: {err_msg}")

        data = response.json()

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=self.ollama_model,
            provider="ollama",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

    # ─── Status ────────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Check status of all providers."""
        self.ollama_available = self._check_ollama()
        status = {
            "gemini": {"available": self.gemini_available, "model": "gemini-2.5-flash"},
            "ollama": {"available": self.ollama_available},
        }
        if status["ollama"]["available"]:
            status["ollama"]["model"] = self.ollama_model
            status["ollama"]["url"] = self.ollama_base_url
        return status

    def get_active_provider(self) -> str:
        """Return which provider will handle the next request."""
        if self.gemini_available:
            return "gemini"
        if self.ollama_available:
            return "ollama"
        return "none"

    def get_model_context_size(self) -> int:
        """Get the context window size for the active model."""
        if self.gemini_available:
            # Gemini 2.5 Flash supports 1M tokens, but we cap at a practical limit
            return 1_000_000

        if self.ollama_available:
            try:
                r = self._http_client.post(
                    f"{self.ollama_base_url}/api/show",
                    json={"name": self.ollama_model},
                    timeout=5.0,
                )
                if r.status_code == 200:
                    data = r.json()
                    # Parse num_ctx from model parameters
                    params = data.get("parameters", "")
                    if isinstance(params, str):
                        for line in params.split("\n"):
                            if "num_ctx" in line:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    return int(parts[-1])
                    # Check modelfile for num_ctx
                    modelfile = data.get("modelfile", "")
                    if "num_ctx" in modelfile:
                        for line in modelfile.split("\n"):
                            if "num_ctx" in line:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    try:
                                        return int(parts[-1])
                                    except ValueError:
                                        pass
            except Exception as e:
                logger.warning(f"Could not detect model context size: {e}")

        # Default fallback for most Ollama models
        return 4096

    def close(self):
        """Clean up HTTP client. Don't kill Ollama — it may be used by others."""
        self._http_client.close()
