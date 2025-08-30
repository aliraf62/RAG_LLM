"""Prompt construction utilities for retrieved documents.

Builds formatted prompts and citation text from retrieved document metadata and content.
"""
from __future__ import annotations

import logging
import re
from typing import List

from core.config.settings import settings
from core.utils.models import RetrievedDocument
from core.utils.patterns import HTML_PATTERN

logger = logging.getLogger(__name__)

# Get metadata field mappings from config
METADATA_FIELDS = settings.get("METADATA_FIELDS", {})

def build_context_prompt(
        question: str,
        primary_domain: str,
        docs: List[RetrievedDocument],
        include_images: bool = True
) -> str:
    """Build a context prompt from retrieved documents.

    Args:
        question: User question
        primary_domain: Classified domain for the question
        docs: Retrieved documents
        include_images: Whether to include images in the prompt

    Returns:
        Formatted prompt with context
    """
    prompt_lines = [f"Domain: {primary_domain}", f"Question: {question}", "", "Context:"]

    for doc in docs:
        # Add images from metadata if present and enabled
        if include_images and settings.get("INCLUDE_IMAGES", True) and doc.metadata:
            # Use standardized field names from config
            images_field = METADATA_FIELDS.get("IMAGES_FIELD", "images")
            asset_type_field = METADATA_FIELDS.get("ASSET_TYPE_FIELD", "asset_type")
            asset_path_field = METADATA_FIELDS.get("ASSET_PATH_FIELD", "asset_path")
            caption_field = METADATA_FIELDS.get("CAPTION_FIELD", "caption")

            # Handle explicit image list
            images = doc.metadata.get(images_field, [])
            if images:
                for img_path in images:
                    caption = doc.metadata.get(caption_field, "") or img_path.split("/")[-1]
                    prompt_lines.append(f"![{caption}]({img_path})")

            # Handle single image from asset_path
            elif doc.metadata.get(asset_type_field) == "image" and doc.metadata.get(asset_path_field):
                img_path = doc.metadata.get(asset_path_field)
                caption = doc.metadata.get(caption_field, "") or img_path.split("/")[-1]
                prompt_lines.append(f"![{caption}]({img_path})")

        # Add text context with source link
        url_field = METADATA_FIELDS.get("URL_FIELD", "url")
        source_url = doc.metadata.get(url_field, "#") if doc.metadata else "#"
        src = f"[Source]({source_url})"

        text = doc.content
        if text:
            # Remove HTML tags if present
            if HTML_PATTERN.search(text):
                text = re.sub(r'<[^>]+>', '', text)
            prompt_lines.append(f"{src}\n{text}")

    return "\n\n".join(prompt_lines)


def get_citation_text(docs: List[RetrievedDocument]) -> str:
    """Generate citation footer text from documents.

    Args:
        docs: Retrieved documents

    Returns:
        Formatted citation text, or empty string if citations are disabled
    """
    if not settings.get("ENABLE_CITATIONS", True):
        return ""

    # Use standardized field names from config
    name_fields = [
        METADATA_FIELDS.get("RATING_NAME_FIELD", "rating-name"),
        METADATA_FIELDS.get("GUIDE_NAME_FIELD", "guide-name"),
        METADATA_FIELDS.get("TITLE_FIELD", "title"),
        METADATA_FIELDS.get("SOURCE_FIELD", "source")
    ]

    printed = set()
    citations = []

    for doc in docs:
        if not doc.metadata:
            # Use source field from RetrievedDocument if metadata is empty
            if doc.source and doc.source not in printed:
                citations.append(f"- {doc.source}")
                printed.add(doc.source)
            continue

        # Try each possible name field in order of preference
        name = None
        for field in name_fields:
            if field in doc.metadata:
                name = doc.metadata.get(field)
                if name:
                    break

        # Fall back to document source if metadata doesn't have a name
        if not name and doc.source:
            name = doc.source

        if name and name not in printed:
            citations.append(f"- {name}")
            printed.add(name)

    if not citations:
        return ""

    return "\n**Cited guides:**\n" + "\n".join(citations)