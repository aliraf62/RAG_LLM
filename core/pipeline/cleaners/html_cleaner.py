"""
pipeline/cleaners/html_cleaner.py

Cleaner that strips unwanted HTML tags and returns plain text.
"""
from __future__ import annotations
from typing import Optional, List
from bs4 import BeautifulSoup

from core.utils.component_registry import register
from core.config.settings import settings
from .base import BaseCleaner
from core.pipeline.base import Row

@register("cleaner", "html")
class HTMLCleaner(BaseCleaner):
    """
    Remove unwanted HTML tags (script, style, nav, footer, etc.)
    and return the page's visible text.
    """
    def __init__(
        self,
        customer_id: Optional[str] = None,
        remove_tags: List[str] = None,
        **config
    ) -> None:
        super().__init__(customer_id=customer_id, **config)
        self.remove_tags = remove_tags or settings.get(
            "HTML_CLEANER_REMOVE_TAGS", []
        )

    def clean_text(self, text: str) -> str:
        """
        Parse HTML, remove unwanted tags, and return clean text.

        Parameters
        ----------
        text : str
            HTML text to clean

        Returns
        -------
        str
            Cleaned plain text
        """
        soup = BeautifulSoup(text, "html.parser")
        for tag in self.remove_tags:
            for element in soup.find_all(tag):
                element.decompose()
        raw = soup.get_text(separator=" ")
        cleaned = " ".join(raw.split())
        return cleaned

