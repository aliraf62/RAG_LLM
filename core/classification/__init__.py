"""
Document classification and tagging module.

This module provides utilities for classifying documents,
extracting relevant tags, and enriching document metadata.
"""

from core.classification.classifier import (
    DocumentClassifier,
    get_document_classifier,
    enrich_document_metadata
)

__all__ = [
    'DocumentClassifier',
    'get_document_classifier',
    'enrich_document_metadata'
]
