"""
ARIA — Model Router

Routes each task to the best available model across all providers.
- Groq: fast chat, research, simple questions, code review
- NVIDIA NIM: reasoning, coding, architecture, deep analysis
- OpenRouter: any model (Claude, GPT, Gemini), fallback for complex tasks

Smart fallback: if preferred provider is down, try the next best.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import AriaConfig, DEFAULT_MODELS, get_available_providers, get_all_known_models
from .llm import (
    LLMClient,
    create_client,
    create_client_for_model,
    detect_provider_from_model,
    get_model_display_name,
    list_known_models,
)


# ─── Task-to-Model Mapping ────────────────────────────────────────────────

# Task types that benefit from specific providers
TASK_ROUTING: Dict[str, str] = {
    # Fast & Cheap → Groq
    "chat": "groq",
    "simple_query": "groq",
    "explain": "groq",
    "documentation": "groq",

    # Balanced → OpenRouter (can use any model)
    "research": "openrouter",
    "code_review": "openrouter",
    "planning": "openrouter",
    "analysis": "openrouter",
    "quick_code": "openrouter",

    # Heavy Reasoning → NVIDIA (free strong models)
    "complex_code": "nvidia",
    "architecture": "nvidia",
    "deep_reasoning": "nvidia",
    "bug_fixing": "nvidia",
    "optimization": "nvidia",
    "ml_model_code": "nvidia",
    "orchestration": "nvidia",
}


def classify_task(task_type: str) -> str:
    """Return the preferred provider for a given task type."""
    return TASK_ROUTING.get(task_type, "groq")


class ModelRouter:
    """
    Routes tasks to the best available model provider.

    Supports:
    - Auto-routing by task type
    - Manual model override
    - Smart fallback if preferred provider is unavailable

    Usage:
        router = ModelRouter(config)
        client = router.get_client("chat")
        client = router.get_client("complex_code")
        client = router.get_client("chat", force_model="claude-3.5-sonnet")
    """

    def __init__(self, config: AriaConfig):
        self.config = config
        self._model_override: Optional[str] = None  # Set via 'model [name]' command

    @property
    def active_model_name(self) -> str:
        """Return the currently active model name."""
        if self._model_override:
            return self._model_override
        return DEFAULT_MODELS.get(self.config.primary_provider, list(DEFAULT_MODELS.values())[0])

    @property
    def active_provider(self) -> str:
        """Return the provider for the currently active model."""
        if self._model_override:
            return detect_provider_from_model(self._model_override)
        return self.config.primary_provider if self.config.primary_provider != "auto" else "groq"

    def set_model(self, model_name: str) -> str:
        """Override the active model. Returns a status message."""
        provider = detect_provider_from_model(model_name)
        available = get_available_providers(self.config)

        # Check if the provider for this model is available
        if provider not in available:
            avail_str = ", ".join(available) if available else "none"
            return (f"Cannot use '{model_name}' — {provider} is not configured. "
                    f"Available: {avail_str}")

        self._model_override = model_name
        display = get_model_display_name(model_name)
        return f"Model set to **{model_name}** ({display})"

    def clear_override(self) -> str:
        """Clear model override, return to auto-routing."""
        self._model_override = None
        return "Model set to **auto** — ARIA will route by task type."

    def get_client(
        self,
        task_type: str = "chat",
        model_override: Optional[str] = None,
    ) -> Tuple[LLMClient, str]:
        """
        Returns (client, provider_name) for the given task.

        Resolution order:
        1. model_override (explicit per-call override)
        2. self._model_override (persistent model switch)
        3. classify_task(task_type) — auto-route by task
        4. Fallback to available providers

        Args:
            task_type: Type of task (e.g., "chat", "complex_code")
            model_override: Optional per-call model override

        Returns:
            Tuple of (LLMClient, provider_name)
            provider_name is "groq", "nvidia", or "openrouter"

        Raises:
            RuntimeError: If no providers are available
        """
        available = get_available_providers(self.config)
        if not available:
            raise RuntimeError("No LLM providers available. Check your API keys in .env")

        # Priority 1: Explicit model override (per-call or persistent)
        target_model = model_override or self._model_override
        if target_model:
            provider = detect_provider_from_model(target_model)
            if provider in available:
                try:
                    client = create_client_for_model(self.config, target_model)
                    return client, provider
                except Exception:
                    pass  # Fall through to auto-routing
            # If the overridden provider is unavailable, fall through

        # Priority 2: Auto-route by task type
        preferred = classify_task(task_type)

        # Try preferred provider
        if preferred in available:
            try:
                model = DEFAULT_MODELS.get(preferred)
                client = create_client(self.config, preferred, model=model)
                return client, preferred
            except Exception:
                pass  # Fall through to fallback

        # Priority 3: Try each available provider in order
        for provider in available:
            if provider == preferred:
                continue  # Already tried above
            try:
                model = DEFAULT_MODELS.get(provider)
                client = create_client(self.config, provider, model=model)
                return client, provider
            except Exception:
                continue  # Try next provider

        # If nothing worked, raise
        raise RuntimeError(
            f"No providers available for task '{task_type}'. "
            f"Available: {', '.join(available)}"
        )

    def list_models(self) -> List[Dict[str, str]]:
        """List all available models with their provider and description."""
        available = get_available_providers(self.config)
        models = []

        for provider in available:
            defaults = DEFAULT_MODELS.get(provider, "")
            display = get_model_display_name(defaults)
            is_active = (self.active_model_name == defaults)
            models.append({
                "provider": provider,
                "model": defaults,
                "description": display,
                "default": True,
                "active": is_active,
            })

        # Add known models for each available provider
        all_models = get_all_known_models(self.config)
        for model_name, desc in all_models.items():
            provider = detect_provider_from_model(model_name)
            if provider in available:
                is_active = (self.active_model_name == model_name)
                models.append({
                    "provider": provider,
                    "model": model_name,
                    "description": desc,
                    "default": False,
                    "active": is_active,
                })

        return models
