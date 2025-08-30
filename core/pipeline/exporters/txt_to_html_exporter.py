"""
pipeline/exporters/txt_to_html_exporter.py

Exporter that reads lines or paragraphs of plain text and writes HTML.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

from core.pipeline.base import Row
from core.utils.component_registry import register
from .base import BaseExporter

@register("exporter", "txt_to_html")
class TxtToHTMLExporter(BaseExporter):
    """
    Export plain text (one row per paragraph) into HTML files.
    """

    def run(self, rows: Iterable[Row]) -> Path:
        rows = self._apply_limit(rows)
        for i, row in enumerate(rows, 1):
            paragraphs = row.text.split("\n\n")
            html = "<html><body>\n"
            for p in paragraphs:
                html += f"<p>{p}</p>\n"
            html += "</body></html>"
            html = self._add_default_style(html)
            file = self.out_dir / f"{i:04d}.html"
            file.write_text(html, encoding="utf-8")
        return self.out_dir