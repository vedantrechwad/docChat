"""
LLM Router — Gemini-first with optional Ollama fallback.

Simple routing: try Gemini API first. If not configured or fails,
fall back to a local Ollama model (if available).
"""

import os
import logging
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

    Usage:
        router = LLMRouter()
        response = router.generate("Explain quantum physics")
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3",
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self._http_client = httpx.Client(timeout=120.0)

        # Validate the key isn't a placeholder
        if self.gemini_api_key and "your_" in self.gemini_api_key.lower():
            self.gemini_api_key = None

        self.gemini_available = bool(self.gemini_api_key)
        self.ollama_available = self._check_ollama()

        if self.gemini_available:
            logger.info("LLM Router: Gemini API configured (primary)")
        if self.ollama_available:
            logger.info(f"LLM Router: Ollama available at {ollama_base_url} (fallback)")
        if not self.gemini_available and not self.ollama_available:
            logger.warning("LLM Router: No LLM provider available!")

    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            r = self._http_client.get(f"{self.ollama_base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

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
        """Generate using the Google GenAI SDK."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.gemini_api_key)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        response = client.models.generate_content(
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
            },
        }

        response = self._http_client.post(
            f"{self.ollama_base_url}/api/chat", json=payload
        )
        response.raise_for_status()
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

    def health_check(self) -> Dict[str, Any]:
        """Check status of all providers."""
        status = {
            "gemini": {"available": self.gemini_available},
            "ollama": {"available": self._check_ollama()},
        }
        if status["ollama"]["available"]:
            status["ollama"]["model"] = self.ollama_model
            status["ollama"]["url"] = self.ollama_base_url
        return status

    def close(self):
        """Clean up HTTP client."""
        self._http_client.close()
