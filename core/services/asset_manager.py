"""
Asset copy helper that respects core.config.paths.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from core.config.paths import OUTPUTS_DIR


def copy_asset(src: Path, customer_id: str, dataset_id: str) -> Path:
    """
    Copy *src* into the customer’s asset directory if it doesn’t exist.

    Returns
    -------
    Path
        Destination path inside
        ``OUTPUTS_DIR/<customer>/<dataset>/assets/<src.name>``.
    """
    dest = (
        OUTPUTS_DIR
        / customer_id
        / dataset_id
        / "assets"
        / src.name
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
    return dest


# Singleton pattern for asset_manager (for backward compatibility)
class AssetManager:
    def copy_asset(self, src: Path, customer_id: str, dataset_id: str) -> Path:
        return copy_asset(src, customer_id, dataset_id)

asset_manager = AssetManager()