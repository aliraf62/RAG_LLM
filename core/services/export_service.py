# core/export_service.py
"""Service layer for guide exporting operations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from core.utils.i18n import get_message
from core.utils.component_registry import component_message

logger = logging.getLogger(__name__)


def export_guides(
    exporter: str,
    output_dir: Union[str, Path],
    workbook: Optional[str] = None,
    assets: Optional[str] = None,
    limit: Optional[int] = None,
    no_captions: bool = False,
    force: bool = False
) -> Path:
    """
    Business logic for exporting guides to HTML.

    Args:
        exporter: Name of the exporter to use
        output_dir: Output directory for HTML files
        workbook: Path to the data source file (XLSB for CSO, parquet for others)
        assets: Path to the assets directory (for CSO)
        limit: Limit on number of guides to export
        no_captions: If True, disables image captioning
        force: If True, forces regeneration of existing files

    Returns:
        Path: Output directory path

    Raises:
        ValueError: If required parameters are missing
    """
    from cli.commands import run_component

    # Prepare exporter arguments
    exporter_args = {
        "out_dir": Path(output_dir),
        "limit": limit,
        "no_captions": no_captions,
        "force_regenerate": force
    }

    # Add exporter-specific arguments
    if exporter == "cso_html":
        # For CSO exporter, we need workbook and assets
        if not workbook:
            raise ValueError(get_message("export_guides.missing_workbook"))
        if not assets:
            raise ValueError(get_message("export_guides.missing_assets"))

        exporter_args["workbook"] = Path(workbook)
        exporter_args["assets_dir"] = Path(assets)

    elif exporter == "parquet_html":
        # For parquet exporter, workbook parameter is the parquet file
        if not workbook:
            raise ValueError(get_message("export_guides.missing_parquet"))

        exporter_args["parquet_file"] = Path(workbook)

    # Run the component and return the result directory
    return run_component("exporter", exporter, exporter_args)
