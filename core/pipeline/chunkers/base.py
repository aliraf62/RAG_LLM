"""
pipeline/chunkers/base.py

Abstract base class for chunker components.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from core.pipeline.base import Row
from typing import List
from core.pipeline.base import BasePipelineComponent
class BaseChunker(BasePipelineComponent, ABC):
    """
    Abstract base class for all Chunker components.

    Chunkers split cleaned text into smaller pieces (chunks) for embedding.
    """
    CATEGORY = "chunker"

    @abstractmethod
    def chunk(self, row: Row) -> List[Row]:
        """
        Split a Row into a list of Row chunks.

        Parameters
        ----------
        row : Row
            Cleaned Row object to chunk.

        Returns
        -------
        List[Row]
            List of Row chunks.
        """
        ...