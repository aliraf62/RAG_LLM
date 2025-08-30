"""
core.llm.providers.openai
========================

Coupa-gateway-aware OpenAI client implementation.

Provides OpenAI-specific implementations of LLM functionality:
- Chat completions (with streaming)
- Embeddings
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Dict, Final, List, Optional

import openai
import requests
from openai.types.chat import ChatCompletionMessageParam

from core.utils.exceptions import EmbeddingError, LLMError
from core.config.settings import settings
from core.llm.base import BaseLLM
from core.utils.component_registry import register

logger: Final = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
def _get_sand_token() -> str:
    """Get authorization token for SAND API."""
    url = settings.SAND_TOKEN_URL  # type: ignore[attr-defined]
    auth = (settings.CLIENT_ID, settings.CLIENT_SECRET)  # type: ignore[attr-defined]
    data = {"grant_type": "client_credentials", "scope": settings.SCOPE}  # type: ignore[attr-defined]

    logger.debug("Requesting SAND token from %s", url)
    resp = requests.post(url, auth=auth, data=data, timeout=10)
    resp.raise_for_status()
    return resp.json().get("access_token", "")


@lru_cache(maxsize=1)
def _memoised_client() -> openai.Client:
    """Create and memoize OpenAI client with proper authentication."""
    sand_token = _get_sand_token()
    # Add specific tenant headers that must be present for all API calls
    tenant_headers = {
        "Authorization": f"Bearer {sand_token}",
        "X-Tenant-Id": settings.TENANT_ID,              # type: ignore[attr-defined]
        "X-Application-Name": settings.APPLICATION_NAME,  # type: ignore[attr-defined]
        "X-Use-Case": getattr(settings, "USE_CASE", "ai-indexing"),
        "X-Coupa-Tenant": settings.TENANT_ID,           # Add this which the API may be expecting
        "X-Coupa-Application": settings.APPLICATION_NAME  # Add this which the API may be expecting
    }

    logger.debug("Creating OpenAI client with headers: %s", tenant_headers)

    return openai.Client(
        base_url=settings.OPENAI_BASE_URL,  # type: ignore[attr-defined]
        api_key="",
        default_headers=tenant_headers,
    )


def _normalize_history(history: Optional[List[Dict[str, str]]]) -> list:
    """Convert history to OpenAI ChatCompletionMessageParam format."""
    if not history:
        return []
    allowed_roles = {"system", "user", "assistant", "tool"}
    return [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in history if m.get("role") in allowed_roles and m.get("content")
    ]


# --------------------------------------------------------------------------- #
# OpenAI LLM implementation                                                   #
# --------------------------------------------------------------------------- #
@register("llm", "openai")
class OpenAILLM(BaseLLM):
    """
    OpenAI LLM provider implementing BaseLLM interface.

    Provides implementations for:
    - Chat completions with history
    - Streaming completions
    - Embedding generation
    """
    def __init__(self):
        """Initialize with memoized OpenAI client."""
        self._client = _memoised_client()

    def refresh(self):
        """Refresh the client to get a new token."""
        # Clear the memoization cache to force a new client creation
        _memoised_client.cache_clear()
        # Re-initialize with fresh client
        self._client = _memoised_client()

    def generate_chat(
        self,
        system_message: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        model: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate a chat completion response.

        Parameters
        ----------
        system_message : str
            System message that guides the model's behavior
        user_message : str
            User's input message
        history : Optional[List[Dict[str, str]]], optional
            Previous conversation history
        max_tokens : int, optional
            Maximum tokens in response
        temperature : float, optional
            Sampling temperature (0-1), lower is more deterministic
        model : Optional[str], optional
            Override model name, default from settings

        Returns
        -------
        Dict[str, str]
            Response with content key containing generated text

        Raises
        ------
        LLMError
            If the API request fails
        """
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_message}
        ]
        messages.extend(_normalize_history(history))
        messages.append({"role": "user", "content": user_message})
        try:
            start = time.time()
            resp = self._client.chat.completions.create(
                model=model or settings.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            logger.debug("OpenAI chat completed in %.2fs", time.time() - start)
            return {"content": content.strip() if content else ""}
        except Exception as exc:  # noqa: BLE001
            logger.error("Chat error: %s", exc, exc_info=True)
            raise LLMError(f"Chat completion failed: {exc}") from exc

    def stream_chat(
        self,
        system_message: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        model: Optional[str] = None,
    ):
        """
        Stream a chat completion response token by token.

        Parameters
        ----------
        system_message : str
            System message to guide model behavior
        user_message : str
            User message
        history : Optional[List[Dict[str, str]]], optional
            Conversation history
        max_tokens : int, optional
            Maximum tokens to generate
        temperature : float, optional
            Temperature for generation
        model : Optional[str], optional
            Model to use

        Yields
        ------
        str
            Generated tokens one by one

        Raises
        ------
        LLMError
            If the API request fails
        """
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_message}
        ]

        # Add conversation history
        messages.extend(_normalize_history(history))

        # Add the current user message
        messages.append({"role": "user", "content": user_message})

        logger.debug("Streaming chat completion with %s", model or settings.model)
        start_time = time.time()

        # Add retry mechanism for handling transient API errors
        max_retries = 3
        retry_delay = 1.0  # Initial retry delay in seconds

        for attempt in range(max_retries):
            try:
                # Create a streaming completion request
                stream = self._client.chat.completions.create(
                    model=model or settings.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,  # Enable streaming
                )

                # Yield tokens as they arrive
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        yield token

                # If we get here, streaming completed successfully
                break

            except (openai.InternalServerError, openai.APIError, openai.APIConnectionError) as e:
                duration = time.time() - start_time
                if attempt < max_retries - 1:
                    retry_delay *= 1.5  # Exponential backoff
                    logger.warning(
                        "OpenAI API error on attempt %d after %.2fs: %s. Retrying in %.2fs...",
                        attempt + 1, duration, str(e), retry_delay
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(
                        "OpenAI streaming failed after %d attempts (%.2fs): %s",
                        max_retries, duration, str(e), exc_info=True
                    )
                    # Fall back to non-streaming mode as a last resort
                    try:
                        logger.info("Attempting non-streaming fallback...")
                        response = self._client.chat.completions.create(
                            model=model or settings.model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=False
                        )
                        full_response = response.choices[0].message.content
                        yield full_response
                        break
                    except Exception as fallback_error:
                        logger.error("Fallback also failed: %s", str(fallback_error))
                        raise LLMError(f"Failed to get response from LLM after multiple attempts: {e}")

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "OpenAI streaming error after %.2fs: %s",
                    duration, str(e), exc_info=True
                )
                raise LLMError(f"Failed to stream response from LLM: {e}")

        logger.info(
            "OpenAI streaming completed in %.2fs",
            time.time() - start_time
        )

    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Parameters
        ----------
        texts : List[str]
            List of texts to embed
        model : Optional[str], optional
            Override embedding model name

        Returns
        -------
        List[List[float]]
            List of embedding vectors

        Raises
        ------
        EmbeddingError
            If the embedding request fails
        """
        try:
            # Print debug information
            logger.debug("Getting embeddings for %d texts with model %s", len(texts), model or settings.embed_model)
            logger.debug("Headers: TENANT_ID=%s, APPLICATION_NAME=%s",
                        settings.TENANT_ID, settings.APPLICATION_NAME)

            # Create request with explicit headers to ensure tenant info is sent
            resp = self._client.embeddings.create(
                input=texts,
                model=model or settings.embed_model,
                extra_headers={
                    "X-Tenant-Id": settings.TENANT_ID,  # Explicitly set tenant ID
                    "X-Application-Name": settings.APPLICATION_NAME,
                    "X-Use-Case": getattr(settings, "USE_CASE", "ai-indexing"),
                    "X-Coupa-Tenant": settings.TENANT_ID,  # Add additional format
                    "X-Coupa-Application": settings.APPLICATION_NAME  # Add additional format
                }
            )
            vectors = [d.embedding for d in resp.data]
            return vectors
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding error: %s", exc, exc_info=True)
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
