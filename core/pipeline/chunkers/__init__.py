from __future__ import annotations
"""
pipeline/chunkers/__init__.py

Chunkers split cleaned text into chunks for embedding.
All concrete chunkers register themselves under the "chunker" category.
"""

from core.utils.component_registry import (
    register as register_chunker,
    get as _get,
    available as _available,
)

from .base import BaseChunker
# Auto-register concrete implementations on import:
from .html_chunker import HTMLChunker  # noqa: F401
from .markdown_chunker import MarkdownChunker  # noqa: F401
from .text_chunker import TextChunker  # noqa: F401

from typing import Type, Tuple

__all__ = [
    "BaseChunker",
    "register_chunker",
    "get_chunker",
    "list_available_chunkers",
    "create_chunker",
]


def get_chunker(name: str) -> Type[BaseChunker]:
    """
    Retrieve the chunker *class* registered under `name`.

    Parameters
    ----------
    name : str
        Registry key of the chunker (e.g. "html", "markdown", "text").

    Returns
    -------
    Type[BaseChunker]
        The chunker class itself.
    """
    return _get("chunker", name)  # type: ignore[return-value]


def list_available_chunkers() -> Tuple[str, ...]:
    """
    List all registered chunker names.

    Returns
    -------
    Tuple[str, ...]
        Registry keys of all available chunkers.
    """
    return _available("chunker")


def create_chunker(name: str, **cfg) -> BaseChunker:
    """
    Instantiate a chunker by its registry name, passing any config.

    Parameters
    ----------
    name : str
        Registry key of the chunker.
    **cfg
        Keyword args forwarded to the chunker's __init__.

    Returns
    -------
    BaseChunker
        An instance of the requested chunker.
    """
    ChunkerCls = get_chunker(name)
    return ChunkerCls(**cfg)