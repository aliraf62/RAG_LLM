# pipeline/loaders/html_fs_loader.py
"""
HTML file system loader
=======================

Walks a directory, reads every `*.html`, extracts the `<meta>` tags,
and returns one *un-chunked* Document per file.

Chunking is left to `pipeline.chunkers.html_chunker`.
"""
from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, Optional, List

from core.config.settings import settings
from core.utils.patterns import HTML_META_TAG_PATTERN
from core.pipeline.cleaners.html_cleaner import HTMLCleaner
from .base import BaseLoader
from core.utils.component_registry import register

logger = logging.getLogger(__name__)

@register("loader", "html")
class HTMLFSLoader(BaseLoader):
    """
    Load HTML files from a directory.

    Processes each HTML file and extracts metadata from meta tags.
    """

    def __init__(self,
                 root_dir: str | Path,
                 recursive: bool = False,
                 cleaner: Optional[HTMLCleaner] = None,
                 **kwargs):
        """
        Initialize the HTML loader.

        Args:
            root_dir: Directory containing HTML files
            recursive: Whether to search subdirectories
            cleaner: Optional custom HTML cleaner

        Raises:
            ValueError: If directory not found
        """
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        self.recursive = recursive
        self.cleaner = cleaner or HTMLCleaner()

        if not self.root_dir.exists():
            logger.error(f"HTML directory not found: {self.root_dir}")
            raise ValueError(f"HTML directory not found: {self.root_dir}")

        logger.debug(f"Initialized HTMLFSLoader for {self.root_dir} (recursive={recursive})")

    # -----------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------
    def _iter_sources(self) -> Iterator[Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]]:
        """
        Iterate over HTML files, extracting text and metadata.

        Returns:
            Iterator of tuples: (file_path, cleaned_text, metadata, structured_data, assets)
        """
        pattern = "**/*.html" if self.recursive else "*.html"
        file_count = 0

        for html_path in sorted(self.root_dir.glob(pattern)):
            try:
                file_count += 1
                logger.debug(f"Processing HTML file: {html_path}")

                raw_html = html_path.read_text(encoding="utf-8")

                # Process with HTMLCleaner to get cleaned content and metadata
                cleaned_data = self.cleaner.clean_for_rag(
                    raw_html,
                    preserve_tables=settings.get("HTML_CLEANER_PRESERVE_TABLES", True),
                    extract_metadata=settings.get("HTML_CLEANER_EXTRACT_METADATA", True),
                    preserve_heading_hierarchy=settings.get("HTML_CLEANER_PRESERVE_HEADING_HIERARCHY", True)
                )

                # Extract basic metadata from HTML file using patterns
                base_meta: Dict[str, Any] = {
                    k: html.unescape(v) for k, v in HTML_META_TAG_PATTERN.findall(raw_html)
                }

                # Combine file metadata with cleaner-extracted metadata
                meta = {**base_meta, **cleaned_data.get("metadata", {}), "file_name": html_path.name,
                        "source": str(html_path), "format": "html"}

                # Extract structured data from cleaner output if available
                structured_data = cleaned_data.get("structured", {})

                # Extract assets from cleaner output if available
                assets = cleaned_data.get("assets", [])

                yield html_path, cleaned_data["text"], meta, structured_data, assets

            except Exception as e:
                logger.error(f"Error processing {html_path}: {e}")
                # Yield a minimal valid result to avoid breaking the pipeline
                yield html_path, f"Error processing file: {e}", {"error": str(e), "file_name": html_path.name}, {}, []

        logger.info(f"Processed {file_count} HTML files from {self.root_dir}")

