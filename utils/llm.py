"""
ARIA — LLM Client Module

Unified interface for three cloud providers:
  1. GroqClient — fast chat via Groq API (llama-3.3-70b)
  2. NVIDIAClient — reasoning/coding via NVIDIA NIM (multiple free models)
  3. OpenRouterClient — any model via OpenRouter (Claude, GPT, Gemini, etc.)

All use OpenAI-compatible APIs with streaming support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Generator, List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import AriaConfig, DEFAULT_MODELS, NVIDIA_FREE_MODELS, GROQ_MODELS, OPENROUTER_COMMON_FREE


# ─── Provider Config ────────────────────────────────────────────────────────

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Map provider names to their default models
PROVIDER_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",
    "openrouter": "google/gemini-2.0-flash:free",
}


# ─── Base Client ────────────────────────────────────────────────────────────

class LLMClient(ABC):
    """Abstract base for all LLM clients."""

    model: str
    provider: str

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming response, yielding tokens one by one."""
        ...

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a complete response (non-streaming)."""
        tokens = list(self.generate_stream(prompt, system_prompt, temperature, max_tokens))
        return "".join(tokens)


# ─── Groq Client ────────────────────────────────────────────────────────────

class GroqClient(LLMClient):
    """Client for Groq API — fast inference on open models."""

    def __init__(self, config: AriaConfig, model: str = "llama-3.3-70b-versatile"):
        self.config = config
        self.model = model
        self.provider = "groq"
        self._client = OpenAI(
            api_key=config.groq_api_key,
            base_url=GROQ_BASE_URL,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            response = self._client.chat.completions.create(**kwargs)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception:
            raise


# ─── NVIDIA NIM Client ──────────────────────────────────────────────────────

class NVIDIAClient(LLMClient):
    """Client for NVIDIA NIM API — multiple free models for reasoning & coding."""

    def __init__(self, config: AriaConfig, model: str = "nvidia/llama-3.3-70b-instruct"):
        self.config = config
        self.model = model
        self.provider = "nvidia"
        self._client = OpenAI(
            api_key=config.nvidia_api_key,
            base_url=NVIDIA_BASE_URL,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            response = self._client.chat.completions.create(**kwargs)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception:
            raise


# ─── OpenRouter Client ──────────────────────────────────────────────────────

class OpenRouterClient(LLMClient):
    """Client for OpenRouter API — access 100+ models via one endpoint.

    Supports any model slug from openrouter.ai/models.
    Free models use the ':free' suffix (e.g., 'google/gemini-2.0-flash:free').
    """

    def __init__(self, config: AriaConfig, model: str = "google/gemini-2.0-flash:free"):
        self.config = config
        self.model = model
        self.provider = "openrouter"
        self._client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        # OpenRouter supports optional headers for app identity
        extra_headers = {
            "HTTP-Referer": "https://github.com/aria-assistant",
            "X-Title": "ARIA Engineering Assistant",
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            response = self._client.chat.completions.create(
                **kwargs,
                extra_headers=extra_headers,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception:
            raise


# ─── Model Utilities ────────────────────────────────────────────────────────

def detect_provider_from_model(model_name: str) -> str:
    """Detect which provider serves a given model name."""
    model_lower = model_name.lower()

    # NVIDIA models
    nvidia_prefixes = ("nvidia/", "deepseek-ai/", "z-ai/", "moonshotai/")
    if any(model_lower.startswith(p) for p in nvidia_prefixes):
        return "nvidia"

    # Groq models (some have / prefixes now: qwen/, meta-llama/, etc.)
    groq_prefixes = ("qwen/", "meta-llama/", "groq/")
    if any(model_lower.startswith(p) for p in groq_prefixes):
        return "groq"

    # Plain model names (no /) are Groq (e.g., "llama-3.3-70b-versatile")
    if "/" not in model_lower:
        return "groq"

    # OpenRouter: anything else with / (e.g., "google/gemini-...", "openai/gpt-...")
    return "openrouter"


def get_model_display_name(model_name: str) -> str:
    """Get a human-readable description for a known model."""
    all_models = {}
    all_models.update(GROQ_MODELS)
    all_models.update(NVIDIA_FREE_MODELS)
    all_models.update(OPENROUTER_COMMON_FREE)
    return all_models.get(model_name, model_name)


def list_known_models() -> Dict[str, str]:
    """Return all known models with provider prefixes for display."""
    models = {}
    for k, v in GROQ_MODELS.items():
        models[f"groq/{k}"] = f"[Groq] {v}"
    for k, v in NVIDIA_FREE_MODELS.items():
        # NVIDIA keys already have 'nvidia/' prefix
        if "/" in k and not k.startswith("groq/") and not k.startswith("nvidia/"):
            models[f"nvidia/{k}"] = f"[NVIDIA] {v}"
        else:
            models[k] = f"[NVIDIA] {v}"
    for k, v in OPENROUTER_COMMON_FREE.items():
        # OpenRouter keys have vendor/model format, avoid double prefix
        models[k] = f"[OpenRouter] {v}"
    return models


# ─── Factory ────────────────────────────────────────────────────────────────

def create_client(
    config: AriaConfig,
    provider: str,
    model: Optional[str] = None,
) -> LLMClient:
    """
    Create an LLM client for the given provider.

    Args:
        config: AriaConfig instance
        provider: "groq", "nvidia", or "openrouter"
        model: Optional model name override

    Returns:
        An LLMClient instance

    Raises:
        ValueError: If provider is unknown or unavailable
    """
    resolved_model = model or PROVIDER_DEFAULT_MODELS.get(provider, "")

    if provider == "groq":
        if not config.groq_available:
            raise ValueError("Groq is not available. Check GROQ_API_KEY in .env")
        return GroqClient(config, resolved_model)

    elif provider == "nvidia":
        if not config.nvidia_available:
            raise ValueError("NVIDIA NIM is not available. Check NVIDIA_API_KEY in .env")
        return NVIDIAClient(config, resolved_model)

    elif provider == "openrouter":
        if not config.openrouter_available:
            raise ValueError("OpenRouter is not available. Check OPENROUTER_API_KEY in .env")
        return OpenRouterClient(config, resolved_model)

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'groq', 'nvidia', or 'openrouter'.")


def create_client_for_model(
    config: AriaConfig,
    model_name: str,
) -> LLMClient:
    """
    Create an LLM client for a specific model name, auto-detecting the provider.

    This is the preferred way to create clients when the user has selected a model.

    Args:
        config: AriaConfig instance
        model_name: Full model name (e.g., "nvidia/llama-3.3-70b-instruct",
                    "llama-3.3-70b-versatile", "google/gemini-2.0-flash:free")

    Returns:
        An LLMClient instance
    """
    provider = detect_provider_from_model(model_name)
    return create_client(config, provider, model=model_name)
