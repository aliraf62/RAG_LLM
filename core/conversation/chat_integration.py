"""
core.conversation.chat_integration
===============================

Integration utilities for chat interfaces and CLI applications.

This module provides high-level functions and classes that make it
easy to integrate conversation functionality with CLI apps and
chat interfaces.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

from core.config.settings import settings
from core.conversation.models import Conversation, Message
from core.conversation.history import save_conversation, get_conversation
from core.conversation.memory import trim_conversation_history
from core.auth.context import get_current_context
from core.utils.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class ChatSession:
    """
    High-level chat session management for CLI and chat applications.

    This class provides a convenient way to manage an ongoing conversation,
    including message history, persistence, and integration with authentication.
    """

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize a new chat session.

        Parameters
        ----------
        conversation_id : str, optional
            ID of an existing conversation to load
        title : str, optional
            Title for a new conversation
        system_prompt : str, optional
            System prompt to use for this conversation
        """
        self.conversation = None
        self.token_limit = settings.get("max_history_tokens", 1024)

        # Try to load an existing conversation
        if conversation_id:
            try:
                self.conversation = get_conversation(conversation_id)
                logger.info(f"Loaded existing conversation: {conversation_id}")
            except AuthenticationError:
                logger.warning("Not authenticated, creating temporary conversation")

        # Create a new conversation if needed
        if self.conversation is None:
            context = get_current_context()
            self.conversation = Conversation(
                id="",  # Will be generated when saved
                user_id=getattr(context, "user_id", "anonymous"),
                customer_id=getattr(context, "customer_id", "default"),
                title=title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                messages=[],
                created_at=datetime.now()
            )

            # Add system prompt if provided
            if system_prompt:
                self.add_message("system", system_prompt)

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a message to the conversation.

        Parameters
        ----------
        role : str
            Message role (system, user, assistant)
        content : str
            Message content
        metadata : Dict, optional
            Additional metadata for the message

        Returns
        -------
        Message
            The added message
        """
        return self.conversation.add_message(role, content, metadata)

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a user message to the conversation."""
        return self.add_message("user", content, metadata)

    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add an assistant message to the conversation."""
        return self.add_message("assistant", content, metadata)

    def save(self) -> Conversation:
        """
        Save the conversation if the user is authenticated.

        Returns
        -------
        Conversation
            The saved conversation

        Raises
        ------
        AuthenticationError
            If no user is authenticated
        """
        return save_conversation(self.conversation)

    def get_messages_for_llm(self, trim_to_token_limit: bool = True) -> List[Dict[str, str]]:
        """
        Get messages formatted for the LLM API.

        Parameters
        ----------
        trim_to_token_limit : bool, optional
            Whether to trim messages to fit token limit

        Returns
        -------
        List[Dict[str, str]]
            Messages formatted for the LLM API
        """
        messages = self.conversation.to_openai_messages()

        if trim_to_token_limit:
            messages = trim_conversation_history(messages, self.token_limit)

        return messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.conversation.to_dict() if self.conversation else {}


class ChatSessionManager:
    """
    Manager for multiple chat sessions.

    This class provides functionality to manage multiple chat sessions
    for a user, including creating, retrieving, and tracking active sessions.
    """

    def __init__(self):
        """Initialize a new chat session manager."""
        self.active_sessions: Dict[str, ChatSession] = {}

    def create_session(
        self,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> ChatSession:
        """
        Create a new chat session.

        Parameters
        ----------
        conversation_id : str, optional
            ID of an existing conversation to load
        title : str, optional
            Title for a new conversation
        system_prompt : str, optional
            System prompt to use for this conversation

        Returns
        -------
        ChatSession
            The created chat session
        """
        session = ChatSession(
            conversation_id=conversation_id,
            title=title,
            system_prompt=system_prompt
        )

        if session.conversation and session.conversation.id:
            self.active_sessions[session.conversation.id] = session

        return session

    def get_session(self, conversation_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by conversation ID.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier

        Returns
        -------
        Optional[ChatSession]
            The chat session if found, None otherwise
        """
        if conversation_id in self.active_sessions:
            return self.active_sessions[conversation_id]

        # Try to load from storage
        session = ChatSession(conversation_id=conversation_id)
        if session.conversation and session.conversation.id:
            self.active_sessions[conversation_id] = session
            return session

        return None

    def close_session(self, conversation_id: str) -> bool:
        """
        Close a chat session.

        Parameters
        ----------
        conversation_id : str
            Conversation identifier

        Returns
        -------
        bool
            True if session was found and closed, False otherwise
        """
        if conversation_id in self.active_sessions:
            try:
                # Try to save before closing
                self.active_sessions[conversation_id].save()
            except Exception as e:
                logger.warning(f"Failed to save session before closing: {e}")

            del self.active_sessions[conversation_id]
            return True

        return False
