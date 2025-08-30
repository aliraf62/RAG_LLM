"""
core/prompt_manager.py
======================

Centralised prompt-template retrieval and formatting.

Responsibilities
----------------
1.  Load prompt templates from the merged *settings* object.
2.  Provide `get_prompt(name)` for one-off constant templates
    (e.g. “SYSTEM”, “RAG_SOURCES_HEADER”).
3.  Offer `PromptManager` for domain/query-aware selection and rendering.
4.  Keep **no** hard-coded strings other than last-ditch fall-backs;
    everything else lives in `core/defaults.py → PROMPTS` or is
    overridden by YAML.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from core.config.defaults import DEFAULT_CONFIG
from core.config.settings import settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Convenience helper (used throughout codebase)                               #
# --------------------------------------------------------------------------- #


def get_prompt(name: str, *, default: str = "") -> str:
    """
    Quick access to constant templates.  Falls back to *default*.
    Tries settings['PROMPTS'], then settings['PROMPT_TEMPLATES'], then DEFAULT_CONFIG["PROMPT_TEMPLATES"].
    """
    # Try PROMPTS in settings
    prompts = settings.get('PROMPTS', {})
    if name in prompts:
        return prompts[name]
    # Try PROMPT_TEMPLATES in settings
    prompt_templates = settings.get('PROMPT_TEMPLATES', {})
    if name in prompt_templates:
        return prompt_templates[name]
    # Try DEFAULT_CONFIG
    if "PROMPT_TEMPLATES" in DEFAULT_CONFIG and name in DEFAULT_CONFIG["PROMPT_TEMPLATES"]:
        return DEFAULT_CONFIG["PROMPT_TEMPLATES"][name]
    # Fallback
    return default


# --------------------------------------------------------------------------- #
# PromptManager class                                                         #
# --------------------------------------------------------------------------- #


class PromptManager:
    """
    Domain-aware, query-aware prompt selector.

    *Template source*: `settings["PROMPTS"]` – a dict::

        PROMPTS:
          SYSTEM: ...
          default: ...
          cso: ...
          cso_How to: ...

    Resolution order (most → least specific):

    1. ``<domain>_<query_type>``
    2. ``<domain>``
    3. ``default``

    ``query_type`` is derived by `_detect_query_type()`.
    """

    _QUERY_REGEX = {
        "How to": re.compile(r"^(how\s+(do\s+i|to))", re.I),
        "Show me": re.compile(r"^(show\s+me|what\s+is)", re.I),
        "Why": re.compile(r"^why\b", re.I),
        "Where": re.compile(r"^where\b", re.I),
    }

    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        # Merge PROMPTS and PROMPT_TEMPLATES from settings, fallback to DEFAULT_CONFIG
        merged = {}
        merged.update(settings.get('PROMPTS', {}))
        merged.update(settings.get('PROMPT_TEMPLATES', {}))
        if "PROMPT_TEMPLATES" in DEFAULT_CONFIG:
            for k, v in DEFAULT_CONFIG["PROMPT_TEMPLATES"].items():
                merged.setdefault(k, v)
        self._templates = merged
        if not self._templates:
            logger.warning("No prompt templates found in configuration or defaults")

    # ------------------------------------------------------------------ #
    # Low-level access                                                   #
    # ------------------------------------------------------------------ #
    def template(self, key: str) -> str:
        """Return template by *key* or hard-coded fallback."""
        if key in self._templates:
            return self._templates[key]

        if "default" in self._templates:
            logger.warning("Template '%s' not found, using default", key)
            return self._templates["default"]

        # absolute fallback – should never happen once config is set
        return (
            "You are a helpful assistant. "
            "Answer the user's question using only the provided context."
        )

    # ------------------------------------------------------------------ #
    # System prompt selection                                            #
    # ------------------------------------------------------------------ #
    def get_system_prompt(self, domain: str = None) -> str:
        """
        Get the system prompt for the specified domain.

        Args:
            domain: The domain to get the system prompt for

        Returns:
            The appropriate system prompt as a string
        """
        # First try domain-specific system prompt
        if domain and f"{domain}" in self._templates:
            return self._templates[f"{domain}"]

        # Then try general SYSTEM prompt
        if "SYSTEM" in self._templates:
            return self._templates["SYSTEM"]

        # Fall back to default prompt
        if "default" in self._templates:
            return self._templates["default"]

        # Last resort fallback
        return (
            "You are a helpful assistant. "
            "Answer the user's question using only the provided context."
        )

    # ------------------------------------------------------------------ #
    # User prompt selection                                              #
    # ------------------------------------------------------------------ #
    def get_user_prompt(self) -> str:
        """
        Get the user prompt template.

        Returns:
            str: The user prompt template to use for RAG queries
        """
        # Look for a USER_PROMPT template in the configuration
        if "USER_PROMPT" in self._templates:
            return self._templates["USER_PROMPT"]

        # Fallback to a standard RAG user prompt if none is defined
        return """Please answer the following question based on the provided context:

Context:
{context}

Question: {question}"""

    # ------------------------------------------------------------------ #
    # Selection logic                                                    #
    # ------------------------------------------------------------------ #
    def _detect_query_type(self, question: str) -> Optional[str]:
        question = question.strip().lower()
        for qtype, rx in self._QUERY_REGEX.items():
            if rx.match(question):
                return qtype
        return None

    def select(self, *, domain: str | None, question: str) -> str:
        """
        Choose the best base template for *domain* + *question*.
        """
        qtype = self._detect_query_type(question)
        if domain and qtype and f"{domain}_{qtype}" in self._templates:
            return self.template(f"{domain}_{qtype}")
        if domain and domain in self._templates:
            return self.template(domain)
        return self.template("default")

    # ------------------------------------------------------------------ #
    # High-level helpers                                                 #
    # ------------------------------------------------------------------ #
    def make_rag_prompt(
        self,
        *,
        question: str,
        context: str,
        domain: str | None = None,
    ) -> str:
        """
        Combine base template + RAG wrapper into one prompt.

        Uses ``settings["RAG_USER_TEMPLATE"]`` for the RAG portion.
        """
        base = self.select(domain=domain, question=question)
        rag_tpl = settings.get(
            "RAG_USER_TEMPLATE",
            "Context:\n{context}\n\nQuestion: {question}",
        )
        rag_part = rag_tpl.format(context=context, question=question)
        return f"{base}\n\n{rag_part}"

    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        Turn retrieved docs into numbered context lines.
        """
        if not docs:
            return "No relevant documents found."

        line_tpl = settings.get(
            "RAG_CONTEXT_FORMAT", "[{index}] {content}"
        )

        lines: List[str] = []
        for i, d in enumerate(docs, 1):
            lines.append(
                line_tpl.format(
                    index=i,
                    content=d.get("content", "")[:2048],  # safety crop
                )
            )
        return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Module-level singleton                                                      #
# --------------------------------------------------------------------------- #
prompt_manager = PromptManager()

__all__ = ["get_prompt", "prompt_manager", "PromptManager"]

