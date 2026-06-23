"""
Orpheus TTS client — proxies to LocalOrpheusTTS OpenAI-compatible API.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]


def strip_for_speech(text: str) -> str:
    """Remove citation markers and markdown for TTS."""
    text = re.sub(r"\[(\d+)\]", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class OrpheusTTSClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_voice: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.base_url = (base_url or os.getenv("ORPHEUS_TTS_URL", "http://localhost:5005")).rstrip("/")
        self.default_voice = default_voice or os.getenv("ORPHEUS_VOICE", "tara")
        self._client = httpx.Client(timeout=timeout)

    def health_check(self) -> Dict[str, Any]:
        try:
            r = self._client.get(f"{self.base_url}/health", timeout=3.0)
            if r.status_code == 200:
                return {"available": True, "url": self.base_url, "detail": r.json() if r.headers.get("content-type", "").startswith("application/json") else {}}
        except Exception as e:
            logger.debug(f"Orpheus health check failed: {e}")
        return {"available": False, "url": self.base_url}

    def list_voices(self) -> List[str]:
        try:
            r = self._client.get(f"{self.base_url}/api/voices", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and "voices" in data:
                    voices = data["voices"]
                    if voices and isinstance(voices[0], dict):
                        return [v.get("name", v.get("id", "tara")) for v in voices]
                    return list(voices)
        except Exception:
            pass
        return DEFAULT_VOICES

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """Generate WAV audio bytes from text."""
        clean = strip_for_speech(text)
        if not clean:
            raise ValueError("Empty text after cleaning")

        # Truncate very long text (Orpheus handles chunking but keep reasonable)
        if len(clean) > 8000:
            clean = clean[:8000] + "..."

        payload = {
            "model": "orpheus",
            "input": clean,
            "voice": voice or self.default_voice,
            "response_format": "wav",
            "speed": max(0.5, min(1.5, speed)),
        }

        r = self._client.post(
            f"{self.base_url}/v1/audio/speech",
            json=payload,
        )
        if r.status_code != 200:
            # Try legacy endpoint
            r = self._client.post(
                f"{self.base_url}/speak",
                json={"text": clean, "voice": voice or self.default_voice},
            )
        if r.status_code != 200:
            raise RuntimeError(f"Orpheus TTS error ({r.status_code}): {r.text[:200]}")

        return r.content

    def close(self) -> None:
        self._client.close()
