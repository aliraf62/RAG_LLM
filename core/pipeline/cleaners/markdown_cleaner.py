"""
pipeline/cleaners/markdown_cleaner.py

Cleaner that removes Markdown code fences and inline code.
"""
from __future__ import annotations
import re
from typing import Optional

from core.utils.component_registry import register
from .base import BaseCleaner
from core.pipeline.base import Row

@register("cleaner", "markdown")
class MarkdownCleaner(BaseCleaner):
    """
    Strip out fenced code blocks and backticks, leaving plain text.
    """
    _FENCE_RE = re.compile(r"```.*?```", flags=re.S)
    _INLINE_RE = re.compile(r"`([^`]+)`")

    def __init__(self, customer_id: Optional[str] = None, **config) -> None:
        """
        Initialize the Markdown cleaner.

        Parameters
        ----------
        customer_id : str, optional
            Customer identifier for metadata processing
        **config : dict
            Additional configuration parameters
        """
        super().__init__(customer_id=customer_id, **config)

    def clean_text(self, text: str) -> str:
        """
        Remove ```code``` fences and `inline code` markers from text.

        Parameters
        ----------
        text : str
            Markdown text to clean

        Returns
        -------
        str
            Cleaned text
        """
        cleaned = self._FENCE_RE.sub("", text)
        cleaned = self._INLINE_RE.sub(r"\1", cleaned)
        return cleaned

