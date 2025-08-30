"""
pipeline/exporters/factory.py

Factory for obtaining exporter classes via the component registry.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from core.utils.component_registry import get as get_exporter

def create_exporter(
    name: str,
    params: Dict[str, Any],
    out_dir: Path | None = None,
) -> Any:
    """
    Instantiate an exporter by registry name.

    Parameters
    ----------
    name : str
        Registered exporter key (e.g. 'parquet_html').
    params : Dict[str,Any]
        Keyword args passed to the exporter constructor.
    out_dir : Path | None
        Optional override for output directory.
    """
    ExporterCls = get_exporter("exporter", name)
    return ExporterCls(**params, out_dir=out_dir)