"""
pipeline/chunkers/html_chunker.py

Chunker for HTML-derived text, splitting on word count.
"""
from __future__ import annotations
from typing import List
from core.utils.component_registry import register
from core.config.settings import settings
from .base import BaseChunker
from core.pipeline.base import Row

@register("chunker", "html")
class HTMLChunker(BaseChunker):
    """
    Split HTML-cleaned text into overlapping word-based chunks.
    """
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.get("DEFAULT_CHUNK_SIZE", 800)
        self.chunk_overlap = chunk_overlap or settings.get("DEFAULT_CHUNK_OVERLAP", 100)

    def chunk(self, row: Row) -> List[Row]:
        """
        Break row.text into lists of words of length *chunk_size*, overlapping by
        *chunk_overlap* words. Returns a list of Row chunks.
        """
        words = row.text.split()
        chunks: List[Row] = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i : i + self.chunk_size])
            chunk_row = Row(
                text=chunk_text,
                metadata=row.metadata.copy(),
                structured=row.structured.copy(),
                assets=row.assets.copy(),
                id=row.id
            )
            chunks.append(chunk_row)
        return chunks