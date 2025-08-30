"""
Shared helpers for chunkers.
"""
from __future__ import annotations

from typing import Callable, List

from core.utils.patterns import WORD_TOKENIZATION_PATTERN


def default_tokenizer() -> Callable[[str], List[str]]:
    """Return simple word splitter (keeps punctuation)."""
    return WORD_TOKENIZATION_PATTERN.findall


def word_token_chunks(
    text: str,
    max_tokens: int,
    overlap: int,
    tokenizer: Callable[[str], List[str]] | None = None,
) -> List[str]:
    """Sliding-window split of *text* into overlapping word-count chunks."""
    tok = tokenizer or default_tokenizer()
    tokens = tok(text)
    total = len(tokens)
    if total <= max_tokens:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    while start < total:
        end = min(start + max_tokens, total)
        chunks.append(" ".join(tokens[start:end]).strip())
        start = end - overlap
    return chunks