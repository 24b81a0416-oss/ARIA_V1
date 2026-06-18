"""
ARIA — Configuration Module

Loads API keys from .env and provides typed config access.
Supports Groq, NVIDIA NIM, OpenRouter, and Ollama (local) providers.

Usage:
    config = load_config()
    if config.groq_available: ...
    if config.nvidia_available: ...
    if config.openrouter_available: ...
    if config.ollama_available: ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# ─── Known Free Models ──────────────────────────────────────────────────────

# Models verified working on NVIDIA NIM free API (tested live)
NVIDIA_FREE_MODELS: Dict[str, str] = {
    "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron-3 Ultra 550B — strongest reasoning & coding",
    "nvidia/nemotron-3-super-120b-a12b": "Nemotron-3 Super 120B — long context, agentic tasks",
    "nvidia/llama-3.3-nemotron-super-49b-v1": "Nemotron Super 49B — balanced speed/quality",
    "z-ai/glm-5.1": "GLM-5.1 — Zhipu complex reasoning & analysis",
    "moonshotai/kimi-k2.6": "Kimi K2.6 — Moonshot multimodal reasoning",
    "meta/llama-3.3-70b-instruct": "Llama 3.3 70B — Meta's latest general model",
    "meta/llama-3.1-70b-instruct": "Llama 3.1 70B — reliable general chat & coding",
    "meta/llama-3.1-8b-instruct": "Llama 3.1 8B — fast lightweight tasks",
    "deepseek-ai/deepseek-v4-flash": "DeepSeek V4 Flash — fast reasoning & code",
    "mistralai/mistral-large-3-675b-instruct-2512": "Mistral Large 3 675B — heavy reasoning",
    "mistralai/mistral-small-4-119b-2603": "Mistral Small 4 119B — efficient general chat",
}

# Models available on Groq (listed from API)
GROQ_MODELS: Dict[str, str] = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B — fast general chat & coding",
    "llama-3.1-8b-instant": "Llama 3.1 8B — ultra-fast simple tasks",
    "qwen/qwen3-32b": "Qwen 3 32B — Alibaba general chat",
    "qwen/qwen3.6-27b": "Qwen 3.6 27B — latest Qwen general model",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout 17B — Meta's latest",
}

# Common free models on OpenRouter (user can use any model slug)
OPENROUTER_COMMON_FREE: Dict[str, str] = {
    "openai/gpt-4o-mini:free": "GPT-4o Mini — affordable general chat",
    "google/gemini-2.0-flash:free": "Gemini 2.0 Flash — fast multimodal",
    "deepseek/deepseek-r1:free": "DeepSeek R1 — reasoning & math",
    "deepseek/deepseek-chat:free": "DeepSeek V3 Chat — general purpose",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B — open general model",
    "qwen/qwen-2.5-72b-instruct:free": "Qwen 2.5 72B — Alibaba general",
    "mistralai/mistral-small-24b-instruct:free": "Mistral Small 24B — efficient reasoning",
}

# Default model selection per provider
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",  # Fast chat & general tasks
    "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",  # Strongest reasoning & coding
    "openrouter": "google/gemini-2.0-flash:free",
}


@dataclass
class AriaConfig:
    """Configuration for ARIA — multi-provider support."""

    # API keys (loaded from .env)
    groq_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # Provider settings
    primary_provider: str = "groq"  # "groq", "nvidia", "openrouter", "ollama", or "auto"
    active_model: str = ""  # Currently selected model (empty = auto-route)

    # Derived state
    groq_available: bool = False
    nvidia_available: bool = False
    openrouter_available: bool = False
    ollama_available: bool = False
    any_provider_available: bool = False


def load_config() -> AriaConfig:
    """
    Load configuration from .env file.

    1. Looks for .env in project root
    2. Reads all API keys
    3. Checks which providers are available (including Ollama via HTTP ping)
    4. Returns AriaConfig
    """
    # Load .env from project root
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Also check home directory
    home_env = Path.home() / ".aria" / ".env"
    if home_env.exists():
        load_dotenv(home_env)

    groq_key = os.getenv("GROQ_API_KEY")
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    primary = os.getenv("PRIMARY_PROVIDER", "auto")

    # Check if Ollama is running (local provider)
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_available = False
    try:
        import requests
        resp = requests.get(f"{ollama_base.rstrip('/')}/api/tags", timeout=2)
        ollama_available = resp.status_code == 200
    except Exception:
        ollama_available = False

    config = AriaConfig(
        groq_api_key=groq_key,
        nvidia_api_key=nvidia_key,
        openrouter_api_key=openrouter_key,
        primary_provider=primary.lower() if primary in ("groq", "nvidia", "openrouter", "auto") else "auto",
        groq_available=bool(groq_key),
        nvidia_available=bool(nvidia_key),
        openrouter_available=bool(openrouter_key),
        ollama_available=ollama_available,
        any_provider_available=bool(groq_key) or bool(nvidia_key) or bool(openrouter_key) or ollama_available,
    )

    return config


def print_provider_status(config: AriaConfig) -> None:
    """Print provider availability status."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Provider", style="bold")
    table.add_column("Status")

    groq_status = "[green]✅ Connected[/green]" if config.groq_available else "[yellow]⏸️  No API key[/yellow]"
    nvidia_status = "[green]✅ Connected[/green]" if config.nvidia_available else "[yellow]⏸️  No API key[/yellow]"
    or_status = "[green]✅ Connected[/green]" if config.openrouter_available else "[yellow]⏸️  No API key[/yellow]"
    ollama_status = "[green]✅ Running[/green]" if config.ollama_available else "[yellow]⏸️  Not detected[/yellow]"

    table.add_row("Groq (Cloud)", groq_status)
    table.add_row("NVIDIA NIM (Cloud)", nvidia_status)
    table.add_row("OpenRouter (Cloud)", or_status)
    table.add_row("Ollama (Local)", ollama_status)

    current_model = config.active_model or f"auto ({DEFAULT_MODELS.get(config.primary_provider, 'groq')})"
    table.add_row("Active Model", f"[cyan]{current_model}[/cyan]")

    console.print(table)


def get_available_providers(config: AriaConfig) -> List[str]:
    """Return list of available provider names."""
    providers = []
    if config.groq_available:
        providers.append("groq")
    if config.nvidia_available:
        providers.append("nvidia")
    if config.openrouter_available:
        providers.append("openrouter")
    if config.ollama_available:
        providers.append("ollama")
    return providers


def get_all_known_models(config: AriaConfig) -> Dict[str, str]:
    """Get all known models with descriptions, filtered by available providers."""
    models = {}

    if config.groq_available:
        models.update(GROQ_MODELS)
    if config.nvidia_available:
        models.update(NVIDIA_FREE_MODELS)
    if config.openrouter_available:
        models.update(OPENROUTER_COMMON_FREE)

    return models
