"""
core.conversation
===============

Central module for conversation management and persistence.

This module provides a unified interface for handling conversations,
including models, history management, and memory utilities.
"""

from core.conversation.models import (
    Conversation,
    Message,
    ConversationMetadata
)

from core.conversation.history import (
    ConversationHistory,
    save_conversation,
    get_conversation,
    get_user_conversations,
    delete_conversation
)

from core.conversation.memory import (
    trim_conversation_history,
    estimate_tokens,
    summarize_conversation
)

__all__ = [
    # Models
    'Conversation',
    'Message',
    'ConversationMetadata',

    # History management
    'ConversationHistory',
    'save_conversation',
    'get_conversation',
    'get_user_conversations',
    'delete_conversation',

    # Memory management
    'trim_conversation_history',
    'estimate_tokens',
    'summarize_conversation'
]
