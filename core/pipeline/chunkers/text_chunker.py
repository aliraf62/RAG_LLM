"""
pipeline/chunkers/text_chunker.py

Chunker for plain text, splitting on word count.
"""
from __future__ import annotations
from typing import List
from core.utils.component_registry import register
from core.config.settings import settings
from .base import BaseChunker
from core.pipeline.base import Row

@register("chunker", "text")
class TextChunker(BaseChunker):
    """
    Split plain text into overlapping word-based chunks.
    """
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        cfg = settings
        self.chunk_size = chunk_size or cfg.get("DEFAULT_CHUNK_SIZE", 800)
        self.chunk_overlap = chunk_overlap or cfg.get("DEFAULT_CHUNK_OVERLAP", 100)

    def chunk(self, row: Row) -> List[Row]:
        """
        Word-based splitting identical to HTMLChunker. Returns a list of Row chunks.
        """
        words = row.text.split()
        chunks: List[Row] = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i : i + self.chunk_size])
            chunks.append(Row(
                text=chunk_text,
                metadata=row.metadata.copy(),
                structured=row.structured.copy(),
                assets=row.assets.copy(),
                id=row.id
            ))
        return chunks