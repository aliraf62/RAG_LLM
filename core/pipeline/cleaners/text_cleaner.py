"""
pipeline/cleaners/text_cleaner.py

Simple text normalizer for plain text inputs.
"""
from __future__ import annotations
import re
from typing import Optional

from core.utils.component_registry import register
from core.config.settings import settings 
from .base import BaseCleaner
from core.pipeline.base import Row

@register("cleaner", "text")
class TextCleaner(BaseCleaner):
    """
    Normalize whitespace and newlines in plain text, with optional max length.
    """
    def __init__(self, customer_id: Optional[str] = None, **config) -> None:
        super().__init__(customer_id=customer_id, **config)
        cfg = settings
        self.norm_ws = cfg.get("TEXT_CLEANER_NORMALIZE_WHITESPACE", True)
        self.norm_nl = cfg.get("TEXT_CLEANER_NORMALIZE_NEWLINES", True)
        self.max_length: int | None = cfg.get("TEXT_CLEANER_MAX_LENGTH", None)

    def clean_text(self, text: str) -> str:
        """
        Apply whitespace and newline normalization, then truncate if needed.

        Parameters
        ----------
        text : str
            Raw text to clean

        Returns
        -------
        str
            Cleaned text
        """
        cleaned = text
        if self.norm_ws:
            cleaned = " ".join(cleaned.split())
        if self.norm_nl:
            cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
        if self.max_length is not None:
            cleaned = cleaned[: self.max_length]
        return cleaned
