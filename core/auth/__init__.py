"""
core.auth
=========

Authentication and multi-tenant context management.

Provides a thread-local store for the current authentication context,
including the active customer, user, and session information.
"""

from core.auth.context import (
    AuthContext,
    get_current_context,
    set_current_context,
    clear_current_context,
    customer_context,
    authenticated_context
)

from core.auth.session import (
    Session,
    create_session,
    get_session,
    verify_session,
    invalidate_session,
    restore_session_from_file
)

from core.auth.user import (
    User,
    get_user,
    authenticate_user
)

from core.auth.conversation import (
    save_conversation,
    get_user_conversations,
    get_conversation_history
)

__all__ = [
    'AuthContext',
    'get_current_context',
    'set_current_context',
    'clear_current_context',
    'customer_context',
    'authenticated_context',
    'Session',
    'create_session',
    'get_session',
    'verify_session',
    'invalidate_session',
    'restore_session_from_file',
    'User',
    'get_user',
    'authenticate_user',
    'save_conversation',
    'get_user_conversations',
    'get_conversation_history'
]
