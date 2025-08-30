"""
Public API for pipeline.indexing

This module exports core indexing primitives and stores.
"""
from __future__ import annotations

# core indexing interfaces and implementations
from .base import BaseVectorStore
from .faiss import FAISSVectorStore
from .indexer import build_index_from_documents
from .similarity import COSINE, EUCLIDEAN, DOT

__all__ = [
    "BaseVectorStore",
    "FAISSVectorStore",
    "build_index_from_documents",
    "COSINE",
    "EUCLIDEAN",
    "DOT",
]