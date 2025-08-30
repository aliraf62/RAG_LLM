"""
Retrieval-Augmented Generation functionality.

This package contains components for implementing RAG (Retrieval-Augmented Generation)
including context formatting, conversations, prompt management, and classification.
"""

from core.rag.conversation import chat_completion
from core.rag.classify import classify_question
from core.rag.context_formatter import build_context_prompt, get_citation_text
from core.rag.prompt_manager import prompt_manager

__all__ = [
    'chat_completion',
    'classify_question',
    'build_context_prompt',
    'get_citation_text',
    'prompt_manager'
]
