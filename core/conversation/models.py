"""
core.conversation.models
======================

Data models for conversations across the system.

This module provides standardized data models for representing
conversations, messages, and related metadata throughout the application.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal

@dataclass
class Message:
    """
    Represents a single message in a conversation.

    Compatible with OpenAI message format while adding
    additional metadata fields useful for our application.
    """
    role: Literal["system", "user", "assistant", "function"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation compatible with OpenAI API."""
        result = {
            "role": self.role,
            "content": self.content
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Create a Message from a dictionary."""
        role = data.get("role")
        content = data.get("content", "")
        timestamp = data.get("timestamp")
        metadata = data.get("metadata", {})

        # Parse timestamp if it's a string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now()
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            role=role,
            content=content,
            timestamp=timestamp,
            metadata=metadata
        )


@dataclass
class ConversationMetadata:
    """
    Metadata for a conversation.

    Contains identifying information and summary data
    without the full message history.
    """
    id: str
    user_id: str
    customer_id: str
    title: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    message_count: int = 0
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """
    Represents a complete conversation with metadata and message history.

    This is the primary data model for conversations throughout the system.
    """
    id: str
    user_id: str
    customer_id: str
    title: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a new message to the conversation.

        Parameters
        ----------
        role : str
            The role of the message sender (system, user, assistant)
        content : str
            The content of the message
        metadata : Dict, optional
            Additional metadata for the message

        Returns
        -------
        Message
            The newly created message
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for storage."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "customer_id": self.customer_id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "summary": self.summary,
            "tags": self.tags,
            "metadata": self.metadata
        }

    def to_metadata(self) -> ConversationMetadata:
        """Convert to metadata-only representation."""
        return ConversationMetadata(
            id=self.id,
            user_id=self.user_id,
            customer_id=self.customer_id,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
            message_count=len(self.messages),
            summary=self.summary,
            tags=self.tags,
            metadata=self.metadata
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Conversation:
        """Create a Conversation from a dictionary."""
        messages_data = data.get("messages", [])
        messages = [Message.from_dict(m) if isinstance(m, dict) else m for m in messages_data]

        # Parse timestamps
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif created_at is None:
            created_at = datetime.now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at)
            except ValueError:
                updated_at = None

        return cls(
            id=data.get("id", ""),
            user_id=data.get("user_id", ""),
            customer_id=data.get("customer_id", ""),
            title=data.get("title", ""),
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            summary=data.get("summary"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )

    def to_openai_messages(self) -> List[Dict[str, str]]:
        """Convert messages to the format expected by the OpenAI API."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
