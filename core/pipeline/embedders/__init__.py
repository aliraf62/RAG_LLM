from __future__ import annotations
"""
pipeline/embedders/__init__.py

Embedders convert text chunks into vector representations.
All concrete embedders register themselves under the "embedder" category.
"""

from core.utils.component_registry import (
    register as register_embedder,
    get as _get,
    available as _available,
)

from .base import BaseEmbedder
# Auto-register concrete implementations on import:
# from .<your_embedder> import <YourEmbedder>  # noqa: F401

from typing import Type, Tuple

__all__ = [
    "BaseEmbedder",
    "register_embedder",
    "get_embedder",
    "list_available_embedders",
    "create_embedder",
]


def get_embedder(name: str) -> Type[BaseEmbedder]:
    """
    Retrieve the embedder *class* registered under `name`.

    Parameters
    ----------
    name : str
        Registry key of the embedder.

    Returns
    -------
    Type[BaseEmbedder]
        The embedder class itself.
    """
    return _get("embedder", name)  # type: ignore[return-value]


def list_available_embedders() -> Tuple[str, ...]:
    """
    List all registered embedder names.

    Returns
    -------
    Tuple[str, ...]
        Registry keys of all available embedders.
    """
    return _available("embedder")


def create_embedder(name: str, **cfg) -> BaseEmbedder:
    """
    Instantiate an embedder by its registry name, passing any config.

    Parameters
    ----------
    name : str
        Registry key of the embedder.
    **cfg
        Keyword args forwarded to the embedder's __init__.

    Returns
    -------
    BaseEmbedder
        An instance of the requested embedder.
    """
    embedder_cls = get_embedder(name)
    return embedder_cls(**cfg)