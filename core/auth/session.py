"""
core.auth.session
================

Session management for authenticated users.

Provides functionality to create, retrieve, verify, and invalidate
user sessions across the application.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import os
import secrets
import time
from typing import Dict, Optional
from pathlib import Path

from core.config.paths import project_path, customer_path
from core.utils.singleton_meta import SingletonMeta
from core.utils.exceptions import AuthenticationError, SessionError

logger = logging.getLogger(__name__)

# Constants
SESSION_EXPIRY_DAYS = 30
SESSION_ID_LENGTH = 32


@dataclass
class Session:
    """
    Represents an authenticated user session.

    Sessions associate a user with a token for a specific period
    and are used to authenticate API requests.
    """
    id: str
    user_id: str
    customer_id: str
    created_at: datetime
    expires_at: datetime
    metadata: Dict = None

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict:
        """Convert session to a dictionary for storage."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "customer_id": self.customer_id,
            "created_at": self.created_at.timestamp(),
            "expires_at": self.expires_at.timestamp(),
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Session:
        """Create a Session instance from a dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            customer_id=data["customer_id"],
            created_at=datetime.fromtimestamp(data["created_at"]),
            expires_at=datetime.fromtimestamp(data["expires_at"]),
            metadata=data.get("metadata", {})
        )


class SessionStore(metaclass=SingletonMeta):
    """
    Store and manage user sessions.

    This class handles session persistence, retrieval, and validation.
    For simplicity, sessions are stored in JSON files per customer.
    In a production environment, this would use a database or Redis.
    """

    def __init__(self) -> None:
        """Initialize the session store."""
        self._sessions: Dict[str, Session] = {}
        self._loaded_customers: set = set()

    def create_session(
        self,
        user_id: str,
        customer_id: str,
        expiry_days: int = SESSION_EXPIRY_DAYS,
        metadata: Dict = None
    ) -> Session:
        """
        Create a new session for a user.

        Parameters
        ----------
        user_id : str
            User identifier
        customer_id : str
            Customer identifier
        expiry_days : int, optional
            Number of days until session expires, defaults to SESSION_EXPIRY_DAYS
        metadata : Dict, optional
            Additional session metadata

        Returns
        -------
        Session
            Newly created session
        """
        # Generate a random session ID
        session_id = secrets.token_hex(SESSION_ID_LENGTH // 2)

        # Create the session
        now = datetime.now()
        session = Session(
            id=session_id,
            user_id=user_id,
            customer_id=customer_id,
            created_at=now,
            expires_at=now + timedelta(days=expiry_days),
            metadata=metadata or {}
        )

        # Store the session
        self._sessions[session_id] = session
        self._save_session(session)

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        Optional[Session]
            Session if found and valid, None otherwise
        """
        # Check if it's already loaded in memory
        session = self._sessions.get(session_id)

        # If not in memory, try to load it
        if not session:
            session = self._load_session(session_id)
            if session:
                self._sessions[session_id] = session

        # If session exists but is expired, remove it
        if session and session.is_expired:
            self.invalidate_session(session_id)
            return None

        return session

    def verify_session(self, session_id: str) -> bool:
        """
        Verify if a session is valid.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if session is valid, False otherwise
        """
        session = self.get_session(session_id)
        return session is not None and not session.is_expired

    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a session.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if session was found and invalidated, False otherwise
        """
        session = self._sessions.pop(session_id, None)
        if session:
            self._delete_session(session_id, session.customer_id)
            return True
        return False

    def _save_session(self, session: Session) -> None:
        """
        Save a session to disk.

        Parameters
        ----------
        session : Session
            Session to save
        """
        try:
            # Get the customer's session directory
            sessions_dir = customer_path(session.customer_id, "sessions")
            sessions_dir.mkdir(parents=True, exist_ok=True)

            # Create the session file path
            session_file = sessions_dir / f"{session.id}.json"

            # Write the session data
            with open(session_file, "w") as f:
                json.dump(session.to_dict(), f)

        except Exception as e:
            logger.error(f"Failed to save session {session.id}: {e}")

    def _load_session(self, session_id: str) -> Optional[Session]:
        """
        Load a session from disk.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        Optional[Session]
            Session if found, None otherwise
        """
        # Since we don't know the customer ID, we need to check all customers
        customers_dir = project_path("customers")
        if not customers_dir.exists():
            return None

        # Look for the session file in each customer's sessions directory
        for customer_dir in customers_dir.iterdir():
            if not customer_dir.is_dir():
                continue

            sessions_dir = customer_dir / "sessions"
            if not sessions_dir.exists() or not sessions_dir.is_dir():
                continue

            session_file = sessions_dir / f"{session_id}.json"
            if session_file.exists():
                try:
                    with open(session_file, "r") as f:
                        session_data = json.load(f)
                        return Session.from_dict(session_data)
                except Exception as e:
                    logger.error(f"Failed to load session {session_id}: {e}")

        return None

    def _delete_session(self, session_id: str, customer_id: str) -> None:
        """
        Delete a session file from disk.

        Parameters
        ----------
        session_id : str
            Session identifier
        customer_id : str
            Customer identifier
        """
        try:
            sessions_dir = customer_path(customer_id, "sessions")
            session_file = sessions_dir / f"{session_id}.json"

            if session_file.exists():
                os.remove(session_file)
                logger.info(f"Deleted session {session_id}")
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")


# Singleton instance
_session_store = SessionStore()


def create_session(
    user_id: str,
    customer_id: str,
    expiry_days: int = SESSION_EXPIRY_DAYS,
    metadata: Dict = None
) -> Session:
    """
    Create a new session for a user.

    Parameters
    ----------
    user_id : str
        User identifier
    customer_id : str
        Customer identifier
    expiry_days : int, optional
        Number of days until session expires
    metadata : Dict, optional
        Additional session metadata

    Returns
    -------
    Session
        Newly created session
    """
    return _session_store.create_session(
        user_id=user_id,
        customer_id=customer_id,
        expiry_days=expiry_days,
        metadata=metadata
    )


def get_session(session_id: str) -> Optional[Session]:
    """
    Get a session by ID.

    Parameters
    ----------
    session_id : str
        Session identifier

    Returns
    -------
    Optional[Session]
        Session if found and valid, None otherwise
    """
    return _session_store.get_session(session_id)


def verify_session(session_id: str) -> bool:
    """
    Verify if a session is valid.

    Parameters
    ----------
    session_id : str
        Session identifier

    Returns
    -------
    bool
        True if session is valid, False otherwise
    """
    return _session_store.verify_session(session_id)


def invalidate_session(session_id: str) -> bool:
    """
    Invalidate a session.

    Parameters
    ----------
    session_id : str
        Session identifier

    Returns
    -------
    bool
        True if session was found and invalidated, False otherwise
    """
    return _session_store.invalidate_session(session_id)


def restore_session_from_file() -> Optional[Session]:
    """
    Restore a session from a saved file.

    This function checks for a saved session ID in the user's home directory
    and attempts to restore the session if found and valid.

    Returns
    -------
    Optional[Session]
        Restored session if found and valid, None otherwise
    """
    import json
    import logging
    from pathlib import Path
    from core.auth.context import AuthContext, set_current_context
    from core.config.settings import apply_customer_settings

    logger = logging.getLogger(__name__)
    session_file = Path.home() / ".ai_assistant_session"

    if not session_file.exists():
        return None

    try:
        with open(session_file, "r") as f:
            data = json.load(f)

        session_id = data.get("session_id")
        if not session_id:
            logger.warning("Invalid session data in saved file")
            return None

        # Verify the session is still valid
        session = get_session(session_id)
        if not session:
            logger.info("Saved session has expired or is invalid")
            return None

        # Set the current context
        context = AuthContext(
            customer_id=session.customer_id,
            user_id=session.user_id,
            session_id=session.id
        )
        set_current_context(context)

        # Apply customer settings
        apply_customer_settings(session.customer_id)

        logger.info(f"Restored session for user {session.user_id} (customer: {session.customer_id})")
        return session

    except Exception as e:
        logger.error(f"Failed to restore session: {e}")
        return None

