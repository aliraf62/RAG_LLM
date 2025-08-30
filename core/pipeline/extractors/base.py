"""
pipeline/extractors/base.py

Abstract base class for extractor components.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from core.pipeline.base import Row
from core.pipeline.base import BasePipelineComponent
class BaseExtractor(BasePipelineComponent, ABC):
    """
    Abstract base class for all Extractor components.

    Extractors read raw data sources (Excel, JSON, CSV, etc.) and yield
    normalized rows for downstream processing.
    """
    CATEGORY = "extractor"

    def __init__(self, **config: Any) -> None:
        """
        Store arbitrary config parameters from YAML or factory.
        """
        self.config = config
        from core.config.settings import settings
        self.settings = settings

    @abstractmethod
    def extract_rows(self) -> Iterable[Row]:
        """
        Yield Row objects one by one.

        Returns
        -------
        Iterable[Row]
        """
        ...

    # protected helpers

    def _caption_asset(self, path: Path) -> str | None:
        from core.pipeline.utils.image_captioner import caption_image
        if self.settings.get("CAPTION_ASSETS", False):
            return caption_image(path)
        return None