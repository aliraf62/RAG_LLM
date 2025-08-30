"""
core.rag.conversation
====================

Utilities for chat completions, history trimming, vision helpers, etc.
Relies on the provider-agnostic LLM accessor in :pymod:`core.llm`.

Note: This module now uses the central conversation module for core functionality.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from core.llm import get_llm_client
from core.utils.i18n import get_message
from core.config.settings import settings
from core.conversation.memory import estimate_tokens, trim_conversation_history
from core.conversation.models import Conversation, Message

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Internal: cached LLM provider instance                                      #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _active_llm():
    """Return (and memoise) the default LLM provider."""
    provider_name = getattr(settings, "llm_provider", "openai")
    logger.info("Using '%s' LLM provider", provider_name)
    return get_llm_client(provider_name)


# --------------------------------------------------------------------------- #
# Helper functions - Backward compatibility                                   #
# --------------------------------------------------------------------------- #
# Keep the original function name for backward compatibility
def build_conversation_history(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Trim history so total token count ≤ *max_tokens*. Uses the new implementation."""
    return trim_conversation_history(messages, max_tokens)


# --------------------------------------------------------------------------- #
# Chat completion helpers                                                     #
# --------------------------------------------------------------------------- #
def chat_completion(
    system_prompt: str,
    user_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    context: str = "",
    question: str = "",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    token_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Generate a chat completion with optional streaming support.

    Parameters
    ----------
    system_prompt : str
        System prompt to guide the model's behavior
    user_prompt : str
        User prompt template (will be formatted with context and question)
    history : Optional[List[Dict[str, str]]], optional
        Conversation history
    context : str, optional
        Context information to include in the user message
    question : str, optional
        Question to answer
    model : Optional[str], optional
        Model name to use, defaults to settings
    temperature : Optional[float], optional
        Temperature for generation, defaults to settings
    max_tokens : Optional[int], optional
        Maximum tokens to generate, defaults to settings
    stream : bool, optional
        Whether to stream the response token by token
    token_callback : Optional[Callable[[str], None]], optional
        Callback function to call for each token when streaming

    Returns
    -------
    str
        Generated response text
    """
    llm = _active_llm()

    # Format user message with context and question
    user_message = user_prompt
    if "{context}" in user_message:
        user_message = user_message.replace("{context}", context or "")
    if "{question}" in user_message:
        user_message = user_message.replace("{question}", question or "")

    processed_history = build_conversation_history(history or [])

    try:
        if stream and token_callback:
            # Use streaming completion with token callback
            response = ""
            for token in stream_chat_completion(
                system_message=system_prompt,
                user_message=user_message,
                history=processed_history,
                model=model or settings.model,
                temperature=temperature or settings.default_temperature,
                max_tokens=max_tokens or settings.max_tokens,
            ):
                token_callback(token)
                response += token
            return response
        else:
            # Use regular blocking completion
            result = llm.generate_chat(
                system_message=system_prompt,
                user_message=user_message,
                history=processed_history,
                model=model or settings.model,
                temperature=temperature or settings.default_temperature,
                max_tokens=max_tokens or settings.max_tokens,
            )
            return result.get("content", "")
    except Exception as exc:
        logger.error("Chat completion error: %s", exc, exc_info=True)
        error_message = get_message("error.llm_completion", error=str(exc))
        if token_callback:
            token_callback(error_message)
        return error_message


def stream_chat_completion(
    system_message: str,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Sequence[str]:
    """
    Stream a chat completion response token by token.

    Parameters
    ----------
    system_message : str
        System message to guide the model's behavior
    user_message : str
        User message
    history : Optional[List[Dict[str, str]]], optional
        Conversation history
    model : Optional[str], optional
        Model name to use
    temperature : Optional[float], optional
        Temperature for generation
    max_tokens : Optional[int], optional
        Maximum tokens to generate

    Yields
    ------
    str
        Generated tokens one by one
    """
    llm = _active_llm()

    try:
        for token in llm.stream_chat(
            system_message=system_message,
            user_message=user_message,
            history=build_conversation_history(history or []),
            model=model or settings.model,
            temperature=temperature or settings.default_temperature,
            max_tokens=max_tokens or settings.max_tokens,
        ):
            yield token
    except Exception as exc:
        logger.error("Streaming chat completion error: %s", exc, exc_info=True)
        yield get_message("error.llm_completion", error=str(exc))


# --------------------------------------------------------------------------- #
# Vision helpers                                                              #
# --------------------------------------------------------------------------- #
def is_vision_model(model: str) -> bool:
    """Return ``True`` if *model* supports image inputs."""
    vision_models = {
        "gpt-4-vision",
        "gpt-4-turbo-vision",
        "gpt-4o",
        "gpt-4o-mini",
        settings.vision_model,
    }
    return model.lower() in {m.lower() for m in vision_models if m}


def format_image_content(user_message: str, image_urls: Sequence[str]) -> List[Dict[str, Any]]:
    """Format message + images for multimodal providers."""
    content: List[Dict[str, Any]] = []
    if user_message:
        content.append({"type": "text", "text": user_message})
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
    return content
