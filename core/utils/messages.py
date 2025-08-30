"""
core.utils.messages
==================

Compatibility module for the legacy messages system.

This module re-exports the new i18n functionality to maintain
backward compatibility with existing code.
"""

from core.utils.i18n import get_message, i18n_manager

# For backward compatibility
SUPPORTED_LANGUAGES = i18n_manager.supported_languages
_CURRENT_LANGUAGE = i18n_manager.current_language

# Re-export the functions and variables needed by existing code
__all__ = ["get_message", "SUPPORTED_LANGUAGES", "_CURRENT_LANGUAGE"]
