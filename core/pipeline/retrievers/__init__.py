from __future__ import annotations
"""
pipeline/retrievers/__init__.py

Retrievers pull back Documents by similarity from a vector store.
All concrete retrievers register themselves under the "retriever" category.
"""

from core.utils.component_registry import (
    register as register_retriever,
    get as _get,
    available as _available,
)

from .base import BaseRetriever
# Auto-register concrete implementations on import:
from .faiss_retriever import FAISSRetriever  # noqa: F401

from typing import Type, Tuple

__all__ = [
    "BaseRetriever",
    "register_retriever",
    "get_retriever",
    "list_available_retrievers",
    "create_retriever",
]


def get_retriever(name: str) -> Type[BaseRetriever]:
    """
    Retrieve the retriever *class* registered under `name`.

    Parameters
    ----------
    name : str
        Registry key of the retriever (e.g. "faiss").

    Returns
    -------
    Type[BaseRetriever]
        The retriever class itself.
    """
    return _get("retriever", name)  # type: ignore[return-value]


def list_available_retrievers() -> Tuple[str, ...]:
    """
    List all registered retriever names.

    Returns
    -------
    Tuple[str, ...]
        Registry keys of all available retrievers.
    """
    return _available("retriever")


def create_retriever(name: str, **cfg) -> BaseRetriever:
    """
    Instantiate a retriever by its registry name, passing any config.

    Parameters
    ----------
    name : str
        Registry key of the retriever.
    **cfg
        Keyword args forwarded to the retriever's __init__.

    Returns
    -------
    BaseRetriever
        An instance of the requested retriever.
    """
    RetrieverCls = get_retriever(name)
    return RetrieverCls(**cfg)