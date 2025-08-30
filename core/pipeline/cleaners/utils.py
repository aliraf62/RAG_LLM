"""
pipeline.cleaners.utils
=============================

Reusable text‑ and HTML‑filter helper functions.

Each filter is **pure** (no global side‑effects) so it can be unit‑tested
individually and composed dynamically by cleaners.  The
`FILTER_REGISTRY` mapping enables settings / YAML to decide which
filters run at runtime.
"""
from __future__ import annotations

from typing import Callable, List

from bs4 import BeautifulSoup
from core.utils.patterns import (
    WHITESPACE_NORMALIZATION_PATTERN,
    NEWLINE_NORMALIZATION_PATTERN,
)
from core.config.settings import settings

FilterFn = Callable[[str], str]

# ---------------------------------------------------------------------- #
# Generic text helpers                                                   #
# ---------------------------------------------------------------------- #
def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space."""
    return WHITESPACE_NORMALIZATION_PATTERN.sub(" ", text)


def normalize_newlines(text: str) -> str:
    """Normalize multi-newline sequences, trim leading/trailing."""
    return NEWLINE_NORMALIZATION_PATTERN.sub("\n", text).strip()


def truncate(text: str, max_len: int) -> str:
    """Cut *text* at *max_len* characters if necessary."""
    return text[:max_len] if len(text) > max_len else text


def safe_to_str(val):
    """Convert value to string, handling pandas Series/DataFrame gracefully."""
    import pandas as pd
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return ""
    if val is None:
        return ""
    return str(val)

# ---------------------------------------------------------------------- #
# HTML helpers                                                           #
# ---------------------------------------------------------------------- #
def strip_html_tags(text: str, tags: List[str] | None = None) -> str:
    """
    Remove specified HTML *tags* (and their content) from *text*.

    Tag list resolution order
    1. Explicit *tags* arg
    2. settings.cleaner_html_remove_tags
    3. core.defaults["cleaner_html_remove_tags"]
    """
    default_tags = ["script", "style", "nav", "footer"]
    if tags is None:
        tags = settings.get("cleaner_html_remove_tags", default_tags)
    if tags is None:  # settings key explicitly set to null
        tags = default_tags
    soup = BeautifulSoup(text, "html.parser")
    for tag in tags:
        for el in soup(tag):
            el.decompose()
    return soup.get_text(" ", strip=True)


