"""
core.conversation.history
=======================

Conversation history management and persistence.

Provides functionality to store and retrieve conversation history
for specific users, organized by customer, with authentication integration.
"""

from __future__ import annotations
import json
import logging
import time
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.config.paths import customer_path
from core.auth.context import get_current_context
from core.utils.exceptions import AuthenticationError
from core.conversation.models import Conversation, ConversationMetadata

logger = logging.getLogger(__name__)


class ConversationHistory:
    """
    Class to manage conversation history for users.

    Conversation history is stored per user, organized by customer.
    Each conversation has a unique ID and contains messages with
    timestamps and metadata.
    """

    @staticmethod
    def get_conversation_dir(customer_id: str, user_id: str) -> Path:
        """Get the directory for a user's conversations."""
        # Get the customer's root directory
        customer_root = customer_path(customer_id)
        # Create the conversations directory path
        return customer_root / "conversations" / user_id

    @staticmethod
    def save_conversation(
        conversation: Conversation
    ) -> Conversation:
        """
        Save a conversation for the authenticated user.

        Parameters
        ----------
        conversation : Conversation
            The conversation to save

        Returns
        -------
        Conversation
            The saved conversation with any updates

        Raises
        ------
        AuthenticationError
            If no user is authenticated
        """
        context = get_current_context()
        if not context.is_authenticated:
            raise AuthenticationError("User must be authenticated to save conversations")

        # Set/update user_id and customer_id from auth context
        conversation.user_id = context.user_id
        conversation.customer_id = context.customer_id

        # Update timestamps
        conversation.updated_at = datetime.now()

        # Save to file
        conversation_dir = ConversationHistory.get_conversation_dir(
            conversation.customer_id,
            conversation.user_id
        )
        conversation_dir.mkdir(parents=True, exist_ok=True)

        # Ensure conversation has an ID
        if not conversation.id:
            conversation.id = f"{int(time.time())}_{secrets.token_hex(4)}"

        conversation_file = conversation_dir / f"{conversation.id}.json"
        with open(conversation_file, "w") as f:
            json.dump(conversation.to_dict(), f, indent=2)

        logger.info(f"Saved conversation {conversation.id} for user {conversation.user_id}")
        return conversation

    @staticmethod
    def get_user_conversations(
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ConversationMetadata]:
        """
        Get a list of conversations for the authenticated user.

        Parameters
        ----------
        limit : int, optional
            Maximum number of conversations to return
        offset : int, optional
            Number of conversations to skip

        Returns
        -------
        List[ConversationMetadata]
            List of conversation metadata objects

        Raises
        ------
        AuthenticationError
            If no user is authenticated
        """
        context = get_current_context()
        if not context.is_authenticated:
            raise AuthenticationError("User must be authenticated to retrieve conversations")

        user_id = context.user_id
        customer_id = context.customer_id

        conversation_dir = ConversationHistory.get_conversation_dir(customer_id, user_id)

        if not conversation_dir.exists():
            return []

        # Get all conversation files
        conversation_files = list(conversation_dir.glob("*.json"))

        # Sort by modification time (newest first)
        conversation_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Apply offset and limit
        if offset > 0:
            conversation_files = conversation_files[offset:]
        if limit is not None:
            conversation_files = conversation_files[:limit]

        # Load metadata for each conversation
        conversations_metadata = []
        for file_path in conversation_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    conversation = Conversation.from_dict(data)
                    conversations_metadata.append(conversation.to_metadata())
            except Exception as e:
                logger.error(f"Failed to load conversation from {file_path}: {e}")

        return conversations_metadata

    @staticmethod
    def get_conversation(conversation_id: str) -> Optional[Conversation]:
        """
        Get a conversation by ID.

        Parameters
        ----------
        conversation_id : str
            The ID of the conversation to retrieve

        Returns
        -------
        Optional[Conversation]
            The conversation if found, None otherwise

        Raises
        ------
        AuthenticationError
            If no user is authenticated
        """
        context = get_current_context()
        if not context.is_authenticated:
            raise AuthenticationError("User must be authenticated to retrieve conversations")

        user_id = context.user_id
        customer_id = context.customer_id

        conversation_dir = ConversationHistory.get_conversation_dir(customer_id, user_id)
        conversation_file = conversation_dir / f"{conversation_id}.json"

        if not conversation_file.exists():
            return None

        try:
            with open(conversation_file, "r") as f:
                data = json.load(f)
                return Conversation.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return None

    @staticmethod
    def delete_conversation(conversation_id: str) -> bool:
        """
        Delete a conversation by ID.

        Parameters
        ----------
        conversation_id : str
            The ID of the conversation to delete

        Returns
        -------
        bool
            True if deletion was successful, False otherwise

        Raises
        ------
        AuthenticationError
            If no user is authenticated
        """
        context = get_current_context()
        if not context.is_authenticated:
            raise AuthenticationError("User must be authenticated to delete conversations")

        user_id = context.user_id
        customer_id = context.customer_id

        conversation_dir = ConversationHistory.get_conversation_dir(customer_id, user_id)
        conversation_file = conversation_dir / f"{conversation_id}.json"

        if not conversation_file.exists():
            return False

        try:
            conversation_file.unlink()
            logger.info(f"Deleted conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            return False


# Simplified function aliases for better usability
save_conversation = ConversationHistory.save_conversation
get_conversation = ConversationHistory.get_conversation
get_user_conversations = ConversationHistory.get_user_conversations
delete_conversation = ConversationHistory.delete_conversation
