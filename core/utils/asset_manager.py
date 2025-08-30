"""
Asset management utilities for handling files and images.

This module provides functions for:
- Copying assets to output directories
- Resolving asset paths and ensuring consistent naming
"""

from __future__ import annotations

import logging
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from core.config.settings import settings

logger = logging.getLogger(__name__)

# Track assets that have been processed to avoid duplicates
_processed_assets: Set[str] = set()

def copy_asset(
    source_path: Union[str, Path],
    destination_dir: Union[str, Path],
    new_name: Optional[str] = None
) -> Tuple[Path, bool]:
    """
    Copy an asset file to a destination directory, tracking copies to avoid duplicates.

    Args:
        source_path: Path to the source asset
        destination_dir: Directory to copy the asset to
        new_name: Optional new name for the asset

    Returns:
        Tuple of (destination path, whether it was newly copied)
    """
    source_path = Path(source_path) if isinstance(source_path, str) else source_path
    destination_dir = Path(destination_dir) if isinstance(destination_dir, str) else destination_dir

    if not source_path.exists():
        logger.warning(f"Source asset not found: {source_path}")
        return (None, False)

    # Create destination directory if it doesn't exist
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Determine destination filename
    dest_name = new_name if new_name else source_path.name
    destination_path = destination_dir / dest_name

    # Check if already processed to avoid duplicate copies
    asset_key = f"{source_path}:{destination_path}"
    if asset_key in _processed_assets:
        return (destination_path, False)

    # Copy the file if destination doesn't exist
    if not destination_path.exists():
        shutil.copy2(source_path, destination_path)
        logger.debug(f"Copied asset: {source_path} -> {destination_path}")
        copied = True
    else:
        logger.debug(f"Asset already exists: {destination_path}")
        copied = False

    # Mark as processed
    _processed_assets.add(asset_key)
    return (destination_path, copied)

def safe_filename(name: str, base_id: str = "", content_type: str = "", default_ext: str = "") -> str:
    """
    Generate a safe filename with appropriate extension.

    Args:
        name: Primary filename to use
        base_id: Fallback ID if name is empty
        content_type: MIME type to derive extension
        default_ext: Default extension if content_type is not recognized

    Returns:
        Safe filename with appropriate extension
    """
    name = (name or "").strip()
    if name:
        # Ensure it has an extension
        if "." not in name and content_type:
            guess = mimetypes.guess_extension(content_type.split(";")[0]) or ""
            name = f"{name}{guess}"
        return name

    # Derive from id + mimetype
    ext = mimetypes.guess_extension(content_type.split(";")[0]) if content_type else default_ext
    return f"{base_id}{ext}"

def find_asset_file(base_path: Union[str, Path], name_patterns: List[str]) -> Optional[Path]:
    """
    Find an asset file using multiple name patterns.

    Args:
        base_path: Base directory to search in
        name_patterns: List of glob patterns to try

    Returns:
        Path to the found file or None if not found
    """
    base_path = Path(base_path) if isinstance(base_path, str) else base_path

    for pattern in name_patterns:
        for match in base_path.glob(pattern):
            return match

    return None
