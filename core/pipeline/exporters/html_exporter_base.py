"""
Helper base for exporters that create HTML files + asset sub-dirs.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .base import BaseExporter


class HTMLExporterBase(BaseExporter):
    """Adds helpers common to HTML-producing exporters."""

    @property
    def html_dir(self) -> Path:
        d = self.out_dir / "html"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def images_dir(self) -> Path:
        d = self.html_dir / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def files_dir(self) -> Path:
        d = self.html_dir / "files"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------ #
    def _copy_asset(self, src: Path, dest_dir: Path) -> Path:
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        return dest