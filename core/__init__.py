"""
Core package initialization.

This module re-exports commonly used components and initializes core functionality.
"""
import logging

from core.utils.component_registry import (
    register,
    get,
    available,
)

from core.utils.models import (
    RetrievedDocument,
    RetrievedDocument,
    RAGRequest,
    RAGResponse
)

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config components                                                           #
# --------------------------------------------------------------------------- #
# Re-export config components for backward compatibility
from core.config.settings import settings, apply_customer_settings
from core.config.paths import (
    project_path,
    get_customer_root,  # Using get_customer_root instead of customer_path
    find_project_root
)

# --------------------------------------------------------------------------- #
# Provider-agnostic LLM helpers                                               #
# --------------------------------------------------------------------------- #
from core.llm import get_llm_client
from core.pipeline.embedders import get_embedder

# --------------------------------------------------------------------------- #
# Component registration system                                               #
# --------------------------------------------------------------------------- #
from core.utils.component_registry import register, get, available

# --------------------------------------------------------------------------- #
# Internationalization                                                        #
# --------------------------------------------------------------------------- #
from core.utils.i18n import get_message

