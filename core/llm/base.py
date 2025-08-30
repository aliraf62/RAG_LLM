# core/llm/base.py
"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.

    All providers must inherit from this and register via the component registry.

    Example
    -------
    from core.llm import get_llm_client
    llm = get_llm_client()
    reply = llm.generate_chat("You are helpful", "Hello!")
    """

    CATEGORY = "llm"

    @abstractmethod
    def generate_chat(
        self,
        system_message: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat completion response.

        Args:
            system_message: System prompt
            user_message: User query
            history: Previous conversation messages
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            model: Override model name

        Returns:
            Response dictionary with 'content' key
        """
        ...

    @abstractmethod
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Override embedding model name

        Returns:
            List of embedding vectors
        """
        ...