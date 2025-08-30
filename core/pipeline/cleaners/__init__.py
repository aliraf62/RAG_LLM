from __future__ import annotations
"""
pipeline/cleaners/__init__.py

Cleaners strip unwanted tags/characters from ingested data.
All concrete cleaners register themselves under the "cleaner" category.
"""

from core.utils.component_registry import (
    register as register_cleaner,
    get as _get,
    available as _available,
)

from .base import BaseCleaner
# Auto-register concrete implementations on import:
from .html_cleaner import HTMLCleaner  # noqa: F401
from .markdown_cleaner import MarkdownCleaner  # noqa: F401
from .text_cleaner import TextCleaner  # noqa: F401

from typing import Type, Tuple

__all__ = [
    "BaseCleaner",
    "register_cleaner",
    "get_cleaner",
    "list_available_cleaners",
    "create_cleaner",
]


def get_cleaner(name: str) -> Type[BaseCleaner]:
    """
    Retrieve the cleaner *class* registered under `name`.

    Parameters
    ----------
    name : str
        Registry key of the cleaner (e.g. "html", "markdown", "text").

    Returns
    -------
    Type[BaseCleaner]
        The cleaner class itself.
    """
    return _get("cleaner", name)  # type: ignore[return-value]


def list_available_cleaners() -> Tuple[str, ...]:
    """
    List all registered cleaner names.

    Returns
    -------
    Tuple[str, ...]
        Registry keys of all available cleaners.
    """
    return _available("cleaner")


def create_cleaner(name: str, **cfg) -> BaseCleaner:
    """
    Instantiate a cleaner by its registry name, passing any config.

    Parameters
    ----------
    name : str
        Registry key of the cleaner.
    **cfg
        Keyword args forwarded to the cleaner's __init__.

    Returns
    -------
    BaseCleaner
        An instance of the requested cleaner.
    """
    CleanerCls = get_cleaner(name)
    return CleanerCls(**cfg)