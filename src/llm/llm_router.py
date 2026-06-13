"""
LLM Router - Hybrid local/cloud model routing for Study Companion.

Supports:
- Ollama (local models: llama3, phi3, nomic-embed-text)
- Groq (cloud, generous free tier)
- OpenAI-compatible APIs

The router lets you assign different models to different task types
(e.g., use phi3 for quick tasks, llama3 for complex reasoning).
"""

import logging
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENAI = "openai"
    GEMINI = "gemini"


class TaskType(str, Enum):
    """Different task types that can be routed to different models."""
    CHAT = "chat"                    # General RAG chat
    SUMMARIZE = "summarize"          # Document summarization
    QUIZ_GENERATE = "quiz_generate"  # Quiz/flashcard generation
    EXPAND = "expand"                # Text expansion
    EXPLAIN = "explain"              # ELI5 / explanations
    TRANSLATE = "translate"          # Translation
    GRAMMAR = "grammar"             # Grammar fixing
    MNEMONIC = "mnemonic"           # Mnemonic generation
    GLOSSARY = "glossary"           # Auto-glossary extraction
    STUDY_PLAN = "study_plan"       # Study plan generation
    PODCAST_SCRIPT = "podcast_script"  # Audio study guide script
    CONCEPT_MAP = "concept_map"     # Concept map extraction
    EMBEDDING = "embedding"         # Text embedding generation


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ModelProvider
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 2000
    base_url: str = ""
    api_key: str = ""

    def __post_init__(self):
        if self.provider == ModelProvider.OLLAMA and not self.base_url:
            self.base_url = "http://localhost:11434"
        elif self.provider == ModelProvider.GROQ and not self.base_url:
            self.base_url = "https://api.groq.com/openai/v1"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Optional[Dict[str, Any]] = None


# Default routing: maps task types to preferred model configs
DEFAULT_TASK_ROUTING: Dict[TaskType, str] = {
    TaskType.CHAT: "primary",
    TaskType.SUMMARIZE: "primary",
    TaskType.QUIZ_GENERATE: "primary",
    TaskType.EXPAND: "primary",
    TaskType.EXPLAIN: "fast",
    TaskType.TRANSLATE: "primary",
    TaskType.GRAMMAR: "fast",
    TaskType.MNEMONIC: "fast",
    TaskType.GLOSSARY: "fast",
    TaskType.STUDY_PLAN: "primary",
    TaskType.PODCAST_SCRIPT: "primary",
    TaskType.CONCEPT_MAP: "primary",
}


class LLMRouter:
    """
    Routes LLM requests to the appropriate provider/model based on task type.

    Usage:
        router = LLMRouter()
        router.add_model("primary", ModelConfig(provider=ModelProvider.OLLAMA, model_name="llama3"))
        router.add_model("fast", ModelConfig(provider=ModelProvider.OLLAMA, model_name="phi3:mini"))
        
        response = router.generate("Explain quantum physics", task_type=TaskType.EXPLAIN)
    """

    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.task_routing: Dict[TaskType, str] = dict(DEFAULT_TASK_ROUTING)
        self._http_client = httpx.Client(timeout=120.0)
        logger.info("LLMRouter initialized")

    def add_model(self, name: str, config: ModelConfig):
        """Register a model configuration under a name (e.g., 'primary', 'fast')."""
        self.models[name] = config
        logger.info(f"Registered model '{name}': {config.provider.value}/{config.model_name}")

    def set_task_route(self, task_type: TaskType, model_name: str):
        """Override which model handles a specific task type."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered. Available: {list(self.models.keys())}")
        self.task_routing[task_type] = model_name

    def get_model_for_task(self, task_type: TaskType) -> ModelConfig:
        """Get the model config assigned to a task type."""
        model_name = self.task_routing.get(task_type, "primary")
        if model_name not in self.models:
            # Fallback to any available model
            if self.models:
                model_name = next(iter(self.models))
                logger.warning(f"Task route '{model_name}' not found, falling back to '{model_name}'")
            else:
                raise ValueError("No models registered in the router")
        return self.models[model_name]

    def generate(
        self,
        prompt: str,
        task_type: TaskType = TaskType.CHAT,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a response using the appropriate model for the task type.
        
        Args:
            prompt: The user prompt
            task_type: The type of task (determines which model to use)
            system_prompt: Optional system prompt
            temperature: Override the model's default temperature
            max_tokens: Override the model's default max_tokens
            model_override: Force use of a specific registered model name
        """
        if model_override and model_override in self.models:
            config = self.models[model_override]
        else:
            config = self.get_model_for_task(task_type)

        temp = temperature if temperature is not None else config.temperature
        tokens = max_tokens if max_tokens is not None else config.max_tokens

        logger.info(f"Generating [{task_type.value}] with {config.provider.value}/{config.model_name}")

        if config.provider == ModelProvider.OLLAMA:
            return self._generate_ollama(config, prompt, system_prompt, temp, tokens)
        elif config.provider == ModelProvider.GROQ:
            return self._generate_openai_compatible(config, prompt, system_prompt, temp, tokens)
        elif config.provider == ModelProvider.OPENAI:
            return self._generate_openai_compatible(config, prompt, system_prompt, temp, tokens)
        elif config.provider == ModelProvider.GEMINI:
            return self._generate_gemini(config, prompt, system_prompt, temp, tokens)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    def _generate_ollama(
        self,
        config: ModelConfig,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate using Ollama's native API."""
        url = f"{config.base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            response = self._http_client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            content = data.get("message", {}).get("content", "")
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }

            return LLMResponse(
                content=content,
                model=config.model_name,
                provider=config.provider.value,
                usage=usage,
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama. Is it running? Start with: ollama serve")
            raise ConnectionError(
                f"Cannot connect to Ollama at {config.base_url}. "
                "Make sure Ollama is running (ollama serve)"
            )

    def _generate_openai_compatible(
        self,
        config: ModelConfig,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate using OpenAI-compatible API (works for Groq, OpenAI, etc.)."""
        url = f"{config.base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Content-Type": "application/json",
        }
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        try:
            response = self._http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                model=config.model_name,
                provider=config.provider.value,
                usage=usage,
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"API error: {e.response.status_code} - {e.response.text}")
            raise

    def _generate_gemini(
        self,
        config: ModelConfig,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate using the official Google GenAI SDK."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Please install 'google-genai' to use Gemini models.")

        if not config.api_key:
            raise ValueError("Gemini API key is required.")

        client = genai.Client(api_key=config.api_key)
        
        # Configure generation parameters
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        if system_prompt:
            gen_config.system_instruction = system_prompt

        try:
            response = client.models.generate_content(
                model=config.model_name,
                contents=prompt,
                config=gen_config,
            )
            
            return LLMResponse(
                content=response.text,
                model=config.model_name,
                provider=config.provider.value,
                usage={}, # The SDK usage stats vary, leaving empty for now
                raw_response={"text": response.text},
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def generate_embedding(
        self,
        text: str,
        model_name: Optional[str] = None,
    ) -> List[float]:
        """
        Generate an embedding vector using Ollama's embedding API.
        Falls back to nomic-embed-text if no model specified.
        """
        model = model_name or "nomic-embed-text"
        
        # Check if we have an Ollama model registered
        ollama_config = None
        for config in self.models.values():
            if config.provider == ModelProvider.OLLAMA:
                ollama_config = config
                break

        base_url = ollama_config.base_url if ollama_config else "http://localhost:11434"
        url = f"{base_url}/api/embed"

        payload = {
            "model": model,
            "input": text,
        }

        try:
            response = self._http_client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def list_available_ollama_models(self) -> List[str]:
        """List models available in local Ollama installation."""
        ollama_config = None
        for config in self.models.values():
            if config.provider == ModelProvider.OLLAMA:
                ollama_config = config
                break

        base_url = ollama_config.base_url if ollama_config else "http://localhost:11434"

        try:
            response = self._http_client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check connectivity to all registered providers."""
        status = {}
        for name, config in self.models.items():
            try:
                if config.provider == ModelProvider.OLLAMA:
                    r = self._http_client.get(f"{config.base_url}/api/tags")
                    status[name] = {"ok": r.status_code == 200, "provider": config.provider.value}
                else:
                    status[name] = {"ok": True, "provider": config.provider.value, "note": "API key configured"}
            except Exception as e:
                status[name] = {"ok": False, "provider": config.provider.value, "error": str(e)}
        return status

    def close(self):
        """Clean up HTTP client."""
        self._http_client.close()


def create_default_router(
    ollama_primary: str = "llama3",
    ollama_fast: str = "phi3",
    groq_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    prefer_cloud: bool = False,
    cloud_provider: str = "gemini",
) -> LLMRouter:
    """
    Create an LLMRouter with sensible defaults.
    
    Uses Ollama as primary, with optional Groq/OpenAI fallbacks.
    """
    router = LLMRouter()

    # Primary local model (complex tasks)
    router.add_model("primary", ModelConfig(
        provider=ModelProvider.OLLAMA,
        model_name=ollama_primary,
        temperature=0.1,
        max_tokens=2000,
    ))

    # Fast local model (micro features)
    router.add_model("fast", ModelConfig(
        provider=ModelProvider.OLLAMA,
        model_name=ollama_fast,
        temperature=0.3,
        max_tokens=500,
    ))

    # Optional cloud fallback
    if groq_api_key:
        router.add_model("groq", ModelConfig(
            provider=ModelProvider.GROQ,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2000,
            api_key=groq_api_key,
        ))
        logger.info("Groq cloud model registered")

    if openai_api_key:
        router.add_model("openai", ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000,
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1",
        ))
        logger.info("OpenAI model registered")

    if gemini_api_key:
        router.add_model("gemini", ModelConfig(
            provider=ModelProvider.GEMINI,
            model_name="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=2000,
            api_key=gemini_api_key,
        ))
        logger.info("Gemini model registered")

    preferred_model = cloud_provider if prefer_cloud else "primary"
    if prefer_cloud and preferred_model in router.models:
        for task_type in DEFAULT_TASK_ROUTING:
            if task_type == TaskType.EMBEDDING:
                continue
            router.set_task_route(task_type, preferred_model)
        logger.info(f"Cloud mode enabled using {preferred_model}")
    elif prefer_cloud:
        logger.warning(
            "Cloud mode requested for '%s', but no matching API key is configured. "
            "Using local Ollama models instead.",
            cloud_provider,
        )

    return router
