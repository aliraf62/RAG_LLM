"""
pipeline/exporters/base.py

Abstract base class for exporter components.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Dict, Any

from core.config.settings import settings
from core.config.paths import OUTPUTS_DIR
from core.pipeline.base import Row
from core.pipeline.base import BasePipelineComponent
class BaseExporter(BasePipelineComponent, ABC):
    """
    Base class for all exporters.

    Exporters convert extractor rows into on-disk artifacts (HTML, JSON, etc.).
    """
    CATEGORY = "exporter"

    def __init__(self, out_dir: Path | None = None) -> None:
        self.out_dir = out_dir or OUTPUTS_DIR / "export"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self, rows: Iterable[Row]) -> Path:
        """
        Perform the export.

        Parameters
        ----------
        rows : Iterable[Row]
            Iterable of Row objects to export.

        Returns
        -------
        Path
            The directory where exported files live.
        """
        ...

    def _apply_limit(self, rows: Iterable[Row]) -> Iterable[Row]:
        """
        Slice off if EXPORT_LIMIT is set.
        """
        limit = settings.get("EXPORT_LIMIT")
        if limit and isinstance(limit, int) and limit > 0:
            from itertools import islice
            return islice(rows, limit)
        return rows

    def _add_default_style(self, html: str) -> str:
        """
        Prepend the default HTML style snippet.
        """
        style = settings.get("HTML_DEFAULT_STYLE", "")
        return style + html