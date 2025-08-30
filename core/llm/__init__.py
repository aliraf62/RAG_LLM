"""
core.llm
========

Provider-agnostic entry point for language-model operations.

Example
-------
from core.llm import get_llm_client
llm = get_llm_client()  # uses settings.LLM_PROVIDER (default "openai")
reply = llm.generate_chat("You are helpful", "Hello!")

from core.llm import get_embedder
embedder = get_embedder()
embeddings = embedder.get_embeddings(["text to embed"])
"""

from typing import Optional, Dict, Any, List
from functools import lru_cache

# Import base classes
from core.llm.base import BaseLLM

# Import provider implementations
from core.llm.providers.openai import OpenAILLM

# Import utilities
from core.utils.component_registry import get as _registry_get
from core.config.settings import settings


def get_llm_client(name: Optional[str] = None) -> BaseLLM:
    """
    Return the LLM client for the specified provider.

    Parameters
    ----------
    name : str, optional
        Provider name (defaults to settings.LLM_PROVIDER or "openai")

    Returns
    -------
    BaseLLM
        LLM client instance
    """
    provider = name or getattr(settings, "LLM_PROVIDER", "openai")
    return _registry_get("llm", provider)()


def get_embedder(name: Optional[str] = None) -> BaseLLM:
    """
    Return the embedder client for the specified provider.

    Parameters
    ----------
    name : str, optional
        Provider name (defaults to settings.EMBED_PROVIDER or settings.LLM_PROVIDER or "openai")

    Returns
    -------
    BaseLLM
        LLM client instance with embedding capability
    """
    provider = name or getattr(settings, "EMBED_PROVIDER",
                  getattr(settings, "LLM_PROVIDER", "openai"))
    return _registry_get("llm", provider)()


@lru_cache(maxsize=1)
def _active_llm() -> BaseLLM:
    """Return (and memoize) the default LLM provider."""
    provider_name = getattr(settings, "LLM_PROVIDER", "openai")
    return get_llm_client(provider_name)


def initialize_client() -> None:
    """Instantiate / cache the default LLM provider (currently 'openai')."""
    get_llm_client()  # Nothing else needed


def refresh_client() -> None:
    """Clear cache so next `get_llm_client()` builds a fresh client."""
    provider = get_llm_client()
    # Every provider class must expose `.refresh()` method for cache-clear
    if hasattr(provider, "refresh"):
        provider.refresh()


__all__ = [
    "get_llm_client",
    "get_embedder",
    "initialize_client",
    "refresh_client",
    "BaseLLM"
]
