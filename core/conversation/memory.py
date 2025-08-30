"""
core.conversation.memory
======================

Working memory management for conversations.

Provides utilities for managing conversation state in memory,
including token counting, history trimming, and summarization.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

from core.config.settings import settings
from core.conversation.models import Conversation, Message
from core.llm import get_llm_client

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Parameters
    ----------
    text : str
        The text to estimate tokens for

    Returns
    -------
    int
        Estimated token count
    """
    if not text:
        return 0

    # Use configurable method for token estimation
    if settings.get("TOKEN_COUNT_METHOD", "word") == "word":
        # Word-based estimation (rough approximation)
        return len(text.split())
    else:
        # Character-based estimation (≈4 chars per token)
        return len(text) // 4


def trim_conversation_history(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Trim conversation history to fit within token limit.

    This function keeps the most recent messages while preserving
    the first system message if present. It estimates token count
    and removes older messages until under the token limit.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        List of message dictionaries with 'role' and 'content' keys
    max_tokens : int, optional
        Maximum allowed tokens, defaults to settings.max_history_tokens

    Returns
    -------
    List[Dict[str, str]]
        Trimmed message list
    """
    max_tokens = max_tokens or settings.get("max_history_tokens", 1024)

    if not messages:
        return []

    # Check if we're already under the limit
    total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    if total_tokens <= max_tokens:
        return messages

    # Preserve system messages
    system_messages = [m for m in messages if m.get("role") == "system"]
    system_token_count = sum(estimate_tokens(m.get("content", "")) for m in system_messages)

    # Keep non-system messages
    other_messages = [m for m in messages if m.get("role") != "system"]

    # Calculate remaining tokens for non-system messages
    remaining_tokens = max_tokens - system_token_count

    if remaining_tokens <= 0:
        # If system messages already exceed limit, keep only most recent system message
        logger.warning("System messages already exceed token limit, trimming drastically")
        if system_messages:
            return [system_messages[-1]]
        return []

    # Trim non-system messages from oldest to newest
    trimmed_messages = []
    for message in reversed(other_messages):
        token_count = estimate_tokens(message.get("content", ""))
        if token_count <= remaining_tokens:
            trimmed_messages.insert(0, message)
            remaining_tokens -= token_count
        else:
            break

    # Combine system messages with trimmed non-system messages
    return system_messages + trimmed_messages


def summarize_conversation(conversation: Conversation) -> str:
    """
    Generate a summary of the conversation using the LLM.

    Parameters
    ----------
    conversation : Conversation
        The conversation to summarize

    Returns
    -------
    str
        Summary of the conversation
    """
    if not conversation.messages or len(conversation.messages) < 3:
        # Not enough messages to summarize
        return conversation.title or "Empty conversation"

    try:
        # Get LLM client
        llm = get_llm_client()

        # Create a prompt for summarization
        system_prompt = "You are a helpful assistant tasked with summarizing a conversation. Create a concise summary (1-2 sentences) capturing the main topics discussed."

        # Convert conversation messages to format expected by LLM
        messages = conversation.to_openai_messages()

        # Trim if needed to meet token limits
        max_summary_tokens = settings.get("max_summary_tokens", 2048)
        trimmed_messages = trim_conversation_history(messages, max_summary_tokens - 200)

        # Add the summary request
        summary_messages = [
            {"role": "system", "content": system_prompt},
            *trimmed_messages,
            {"role": "user", "content": "Please summarize our conversation so far in 1-2 sentences."}
        ]

        # Generate summary
        response = llm.chat.completions.create(
            model=settings.get("model", "gpt-4o-mini"),
            messages=summary_messages,
            temperature=0.3,
            max_tokens=100
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        logger.error(f"Failed to generate conversation summary: {e}")
        # Fallback to basic title
        return conversation.title or "Conversation"
