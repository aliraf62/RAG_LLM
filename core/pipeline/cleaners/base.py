"""
pipeline/cleaners/base.py

Abstract base class for cleaner components.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import logging

from core.pipeline.base import Row
from core.pipeline.base import BasePipelineComponent
from core.config.settings import settings
from core.classification.classifier import enrich_document_metadata


class BaseCleaner(BasePipelineComponent, ABC):
    """
    Abstract base class for all Cleaner components.

    Cleaners remove unwanted content (HTML tags, markdown syntax, extra whitespace)
    from raw text before chunking.
    """
    CATEGORY = "cleaner"

    def __init__(self, customer_id: Optional[str] = None, **config):
        """
        Initialize the cleaner.

        Parameters
        ----------
        customer_id : str, optional
            Customer identifier for metadata processing
        **config : dict
            Additional configuration parameters
        """
        super().__init__(**config)
        self.customer_id = customer_id

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Parameters
        ----------
        text : str
            Raw text to clean

        Returns
        -------
        str
            Cleaned text
        """
        ...

    def clean(self, row: Row) -> Row:
        """
        Clean and normalize a Row, removing unwanted content.
        If SMART_CLASSIFY is enabled, also enriches metadata using document classification.

        Parameters
        ----------
        row : Row
            Row object to clean.

        Returns
        -------
        Row
            Cleaned Row object with potentially enriched metadata.
        """
        # Clean the text
        cleaned_text = self.clean_text(row.text)

        # Create new metadata dictionary to avoid modifying the original
        metadata = row.metadata.copy()

        # Debug information about classification settings
        logger = logging.getLogger(__name__)
        logger.info(f"SMART_CLASSIFY setting: {settings.get('SMART_CLASSIFY')}")
        logger.info(f"Customer ID: {self.customer_id}")

        # Enrich metadata with classification if enabled
        if settings.get("SMART_CLASSIFY", False) and self.customer_id:
            logger.info("Applying classification to enrich metadata")
            metadata = enrich_document_metadata(metadata, cleaned_text, self.customer_id)
        else:
            logger.info("Classification skipped: either SMART_CLASSIFY is disabled or customer_id is missing")

        # Create and return a new Row with cleaned text and possibly enriched metadata
        return Row(
            id=row.id,
            text=cleaned_text,
            metadata=metadata,
            structured=row.structured,
            assets=row.assets
        )

