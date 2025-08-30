"""
Generate captions for images via the configured LLM provider.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from core.llm import get_llm_client
from core.config.settings import settings

logger = logging.getLogger(__name__)


def caption_image(image_path: Path) -> Optional[str]:
    """
    Return a one-sentence caption or **None** if disabled.

    Controlled by ``settings.caption_assets``.
    """
    if not settings.get("caption_assets", False):
        return None

    llm = get_llm_client()  # provider chosen via settings
    try:
        prompt = f"Write a short caption for this image file name: {image_path.name}"
        resp = llm.chat(prompt)
        return resp.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Captioning failed for %s: %s", image_path, exc)
        return None