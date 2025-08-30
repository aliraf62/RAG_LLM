"""
pipeline/chunkers/markdown_chunker.py

Chunker for Markdown, preferring header-based splits, with fallback.
"""
from __future__ import annotations
from typing import List
import re
from core.utils.component_registry import register
from core.config.settings import settings
from .base import BaseChunker
from core.pipeline.base import Row

_HEADER_RE = re.compile(r"(^# .+$)", flags=re.MULTILINE)

@register("chunker", "markdown")
class MarkdownChunker(BaseChunker):
    """
    Split on top-level Markdown headers; if none, fallback to word-based chunks.
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
        Attempt header-based splitting; fallback to word-chunks. Returns a list of Row chunks.
        """
        parts = _HEADER_RE.split(row.text)
        chunks: List[Row] = []
        if len(parts) > 1:
            for p in parts:
                p = p.strip()
                if p:
                    chunks.append(Row(
                        text=p,
                        metadata=row.metadata.copy(),
                        structured=row.structured.copy(),
                        assets=row.assets.copy(),
                        id=row.id
                    ))
            return chunks
        # Otherwise fallback to word-based chunking
        words = row.text.split()
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