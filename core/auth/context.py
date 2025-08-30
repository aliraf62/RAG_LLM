"""
core.auth.context
================

Thread-local context for authentication and customer context.

Provides a thread-safe way to manage the current authentication context
(user, customer, session) across the application.
"""

from __future__ import annotations
import logging
import threading
from typing import Dict, Optional, Any, ContextManager
from contextlib import contextmanager

from core.config.settings import settings, apply_customer_settings
from core.services.customer_service import customer_service

logger = logging.getLogger(__name__)

class AuthContext:
    """
    Authentication context containing user and customer information.

    This class holds the current authenticated user, active customer,
    and session details for the current request/operation.
    """

    def __init__(
        self,
        customer_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **metadata: Any
    ) -> None:
        """
        Initialize a new authentication context.

        Parameters
        ----------
        customer_id : str, optional
            Customer identifier to activate
        user_id : str, optional
            User identifier
        session_id : str, optional
            Session identifier
        metadata : dict
            Additional context metadata
        """
        self.customer_id = customer_id
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata

        # Apply customer settings if a customer ID is provided
        if customer_id:
            try:
                apply_customer_settings(customer_id)
                logger.info(f"Applied settings for customer: {customer_id}")
            except Exception as e:
                logger.error(f"Failed to apply customer settings for {customer_id}: {e}")

    @property
    def is_authenticated(self) -> bool:
        """Check if there's an authenticated user."""
        return bool(self.user_id and self.session_id)

    @property
    def has_active_customer(self) -> bool:
        """Check if there's an active customer context."""
        return bool(self.customer_id)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AuthContext(customer_id={self.customer_id!r}, "
            f"user_id={self.user_id!r}, session_id={self.session_id!r})"
        )


# Thread-local storage for the current context
_CONTEXT_LOCAL = threading.local()


def get_current_context() -> AuthContext:
    """
    Get the current authentication context for this thread.

    If no context has been set for the current thread, returns
    an empty context (unauthenticated, no active customer).

    Returns
    -------
    AuthContext
        Current authentication context
    """
    if not hasattr(_CONTEXT_LOCAL, "context"):
        _CONTEXT_LOCAL.context = AuthContext()
    return _CONTEXT_LOCAL.context


def set_current_context(context: AuthContext) -> None:
    """
    Set the current authentication context for this thread.

    Parameters
    ----------
    context : AuthContext
        Authentication context to set
    """
    _CONTEXT_LOCAL.context = context


def clear_current_context() -> None:
    """Clear the current authentication context for this thread."""
    if hasattr(_CONTEXT_LOCAL, "context"):
        delattr(_CONTEXT_LOCAL, "context")


@contextmanager
def customer_context(customer_id: str) -> ContextManager[AuthContext]:
    """
    Context manager for temporarily setting a customer context.

    Preserves the original context and restores it after the block exits.

    Parameters
    ----------
    customer_id : str
        Customer identifier to activate

    Yields
    ------
    AuthContext
        New authentication context with the specified customer

    Examples
    --------
    >>> with customer_context("acme"):
    ...     # Code here runs with acme customer settings
    ...     result = run_pipeline()
    >>> # Original context is restored
    """
    original_context = get_current_context()

    # Create a new context with the specified customer but preserving user/session
    new_context = AuthContext(
        customer_id=customer_id,
        user_id=original_context.user_id,
        session_id=original_context.session_id,
    )

    try:
        # Set the new context
        set_current_context(new_context)
        yield new_context
    finally:
        # Restore the original context
        set_current_context(original_context)


@contextmanager
def authenticated_context(
    user_id: str,
    customer_id: str,
    session_id: str,
    **metadata: Any
) -> ContextManager[AuthContext]:
    """
    Context manager for temporarily setting an authenticated context.

    Preserves the original context and restores it after the block exits.

    Parameters
    ----------
    user_id : str
        User identifier
    customer_id : str
        Customer identifier
    session_id : str
        Session identifier
    metadata : dict
        Additional context metadata

    Yields
    ------
    AuthContext
        New authenticated context

    Examples
    --------
    >>> with authenticated_context("john.doe", "acme", "abc123"):
    ...     # Code here runs as john.doe from acme
    ...     result = run_pipeline_as_user()
    >>> # Original context is restored
    """
    original_context = get_current_context()

    # Create a new fully authenticated context
    new_context = AuthContext(
        customer_id=customer_id,
        user_id=user_id,
        session_id=session_id,
        **metadata
    )

    try:
        # Set the new context
        set_current_context(new_context)
        yield new_context
    finally:
        # Restore the original context
        set_current_context(original_context)
