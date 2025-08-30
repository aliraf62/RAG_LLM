"""
pipeline/exporters/parquet_html_exporter.py

Exporter that reads a Parquet dataset and writes per-row HTML files.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from core.utils.component_registry import register
from .base import BaseExporter
from core.pipeline.base import Row

@register("exporter", "parquet_html")
class ParquetHTMLExporter(BaseExporter):
    """
    Export Parquet rows to individual HTML files.
    Expects each row to have a 'content' field and optional 'title'.
    """

    def __init__(self, parquet_path: Path, out_dir: Path | None = None) -> None:
        super().__init__(out_dir)
        self.parquet_path = parquet_path

    def run(self, rows: Optional[Iterable[Row]] = None) -> Path:
        """
        If `rows` is None, read the entire Parquet file.
        Otherwise export the provided rows iterable.
        """
        if rows is None:
            df = pd.read_parquet(self.parquet_path)
            rows = (
                Row(text=row["content"], metadata={"title": row.get("title", "")}, assets=[])
                for _, row in df.iterrows()
            )

        rows = self._apply_limit(rows)
        for i, row in enumerate(rows, 1):
            html = f"<h1>{row.metadata.get('title','')}</h1>\n" + row.text
            html = self._add_default_style(html)
            file = self.out_dir / f"{i:04d}.html"
            file.write_text(html, encoding="utf-8")
        return self.out_dir