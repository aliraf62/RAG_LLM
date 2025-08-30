"""
pipeline/indexing/base.py

Abstract base class for vector store components.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Sequence

from langchain.schema import Document
from core.pipeline.base import BasePipelineComponent, Row

class BaseVectorStore(BasePipelineComponent, ABC):
    """
    Abstract base class for all vector store backends (FAISS, Weaviate, Qdrant, etc.).

    Provides uniform `build_index`, `append_to_index`, and requires `search()`.
    """
    CATEGORY = "vectorstore"

    def build_index(
        self,
        chunks: Sequence[Dict[str, Any]],
        embed_fn: Callable[[List[str]], List[List[float]]],
        *,
        batch_size: int = 64,
    ) -> None:
        """
        Build a new index from the provided chunk dictionaries.

        Parameters
        ----------
        chunks : Sequence[Dict[str, Any]]
            List of dicts each containing 'text' and 'metadata'.
        embed_fn : Callable
            Function to embed a list of text strings.
        batch_size : int, optional
            Number of texts to embed per batch.
        """
        self._build_or_append(chunks, embed_fn, batch_size, build=True)

    def build_index_from_rows(
        self,
        rows: Sequence[Row],
        embed_fn: Callable[[List[str]], List[List[float]]],
        *,
        batch_size: int = 64,
    ) -> None:
        """
        Build a new index from the provided Row objects.
        """
        dict_chunks = [row.to_dict() for row in rows]
        self.build_index(dict_chunks, embed_fn, batch_size=batch_size)

    def append_to_index(
        self,
        chunks: Sequence[Dict[str, Any]],
        embed_fn: Callable[[List[str]], List[List[float]]],
        *,
        batch_size: int = 64,
    ) -> None:
        """
        Append the provided chunk dictionaries to an existing index.

        Parameters
        ----------
        chunks : Sequence[Dict[str, Any]]
            List of dicts each containing 'text' and 'metadata'.
        embed_fn : Callable
            Function to embed a list of text strings.
        batch_size : int, optional
            Number of texts to embed per batch.
        """
        self._build_or_append(chunks, embed_fn, batch_size, build=False)

    def append_to_index_from_rows(
        self,
        rows: Sequence[Row],
        embed_fn: Callable[[List[str]], List[List[float]]],
        *,
        batch_size: int = 64,
    ) -> None:
        """
        Append the provided Row objects to an existing index.
        """
        dict_chunks = [row.to_dict() for row in rows]
        self.append_to_index(dict_chunks, embed_fn, batch_size=batch_size)

    @abstractmethod
    def _build_or_append(
        self,
        chunks: Sequence[Dict[str, Any]],
        embed_fn: Callable[[List[str]], List[List[float]]],
        batch_size: int,
        *,
        build: bool,
    ) -> None:
        """
        Internal helper to either build a new index (build=True)
        or append to an existing one (build=False).
        """

    @abstractmethod
    def search(self, query: str, k: int) -> List[Document]:
        """
        Search for the top-k Documents most similar to the query.

        Parameters
        ----------
        query : str
            Natural-language query.
        k : int
            Number of nearest neighbors to return.

        Returns
        -------
        List[Document]
            Retrieved documents.
        """
        ...