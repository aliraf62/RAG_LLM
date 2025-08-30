"""
Lightweight text extractor for Excel cells.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List

import pandas as pd


def extract_text_from_excel(path: Path, sheet_name: str | None = None) -> List[str]:
    """
    Return a list of cell strings (row-major) from the Excel file.

    Useful for embedding entire workbooks quickly.
    """
    data = []
    with path.open("rb") as fh:
        buf = BytesIO(fh.read())
    xl = pd.ExcelFile(buf, engine="openpyxl")
    sheets = [sheet_name] if sheet_name else xl.sheet_names
    for name in sheets:
        df = xl.parse(name)
        for _, row in df.iterrows():
            data.extend(str(v) for v in row.tolist() if pd.notna(v))
    return data