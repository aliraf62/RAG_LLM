"""
core.auth.conversation
=====================

DEPRECATED: This module is maintained for backward compatibility only.
Please use core.conversation.history instead for conversation management.
"""

import logging
from typing import Dict, List, Optional, Any

from core.conversation.history import (
    ConversationHistory,
    save_conversation as _save_conversation,
    get_conversation as _get_conversation_history,
    get_user_conversations as _get_user_conversations,
    delete_conversation as _delete_conversation
)
from core.conversation.models import Conversation

logger = logging.getLogger(__name__)
logger.warning(
    "core.auth.conversation is deprecated. "
    "Please use core.conversation.history instead."
)

# Maintain backward compatibility
def save_conversation(messages: List[Dict[str, Any]], title: Optional[str] = None, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    DEPRECATED: Use core.conversation.history.save_conversation instead.

    Save a conversation for the current user.
    """
    from core.conversation.models import Message, Conversation

    # Convert old-style messages to new Conversation object
    conv_messages = []
    for msg in messages:
        conv_messages.append(Message(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            metadata=msg.get("metadata", {})
        ))

    # Create conversation object
    conversation = Conversation(
        id=conversation_id or "",
        user_id="",  # Will be filled from auth context
        customer_id="", # Will be filled from auth context
        title=title or "",
        messages=conv_messages
    )

    # Save using new system
    result = _save_conversation(conversation)

    # Convert back to old format for compatibility
    return result.to_dict()

def get_user_conversations(limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use core.conversation.history.get_user_conversations instead.

    Get a list of conversations for the current user.
    """
    result = _get_user_conversations(limit=limit, offset=offset)
    return [meta.__dict__ for meta in result]

def get_conversation_history(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    DEPRECATED: Use core.conversation.history.get_conversation instead.

    Get full conversation history by ID.
    """
    result = _get_conversation_history(conversation_id)
    return result.to_dict() if result else None

def delete_conversation(conversation_id: str) -> bool:
    """
    DEPRECATED: Use core.conversation.history.delete_conversation instead.

    Delete a conversation by ID.
    """
    return _delete_conversation(conversation_id)

# Export the original class for backward compatibility
ConversationHistory = ConversationHistory
