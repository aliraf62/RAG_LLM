"""
pipeline/embedders/base.py

Abstract base class for embedder components.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, List
from core.pipeline.base import Row, BasePipelineComponent

class BaseEmbedder(BasePipelineComponent, ABC):
    """
    Abstract base class for all Embedder components.

    Embedders take a sequence of text chunks and return their vector embeddings.
    """
    CATEGORY = "embedder"

    @abstractmethod
    def embed_text(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Embed a list of text strings into vectors.

        Parameters
        ----------
        texts : Sequence[str]
            The input text to embed.

        Returns
        -------
        List[List[float]]
            A list of embedding vectors.
        """
        ...

    @abstractmethod
    def embed_rows(self, rows: Sequence[Row]) -> List[List[float]]:
        """
        Embed a list of Row objects into vectors.

        Parameters
        ----------
        rows : Sequence[Row]
            The input Row objects to embed.

        Returns
        -------
        List[List[float]]
            A list of embedding vectors.
        """
        ...