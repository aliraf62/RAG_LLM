# core/indexing_service.py
"""Service layer for index building operations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from core.config.settings import settings
from core.utils.i18n import get_message

logger = logging.getLogger(__name__)

def get_vector_store_path(dataset_name, customer=None):
    """Get the path to the vector store for a dataset."""
    if customer:
        from core.config.paths import customer_vector_store_path
        return customer_vector_store_path(customer, dataset_name)
    else:
        from core.config.paths import project_path
        return project_path("vector_store", dataset_name)

def build_index(
        html_dir: Path,
        output_dir: Path,
        chunk_strategy: Optional[str] = None,
        limit: Optional[int] = None,
        batch_size: Optional[int] = None,
        similarity_algorithm: Optional[str] = None,
) -> Path:
    """
    Business logic for building a search index from HTML files.

    Args:
        html_dir: Directory containing HTML files
        output_dir: Output directory for the index
        chunk_strategy: Strategy for chunking documents
        limit: Maximum number of files to process
        batch_size: Batch size for embeddings
        similarity_algorithm: Algorithm for vector similarity (cosine, dot, Euclidean)

    Returns:
        Path to the created index

    Raises:
        ValueError: If input directory doesn't exist
    """
    # Validate input
    if not html_dir.exists():
        raise ValueError(get_message("build_index.dir_not_found", dir_path=html_dir))

    # Get defaults from config
    if chunk_strategy is None:
        chunk_strategy = settings.get("INDEX_DEFAULTS", {}).get("CHUNK_STRATEGY", "header")

    if batch_size is None:
        batch_size = settings.get("INDEX_DEFAULTS", {}).get("BATCH_SIZE", 64)

    if similarity_algorithm is None:
        similarity_algorithm = settings.get("VECTOR_STORE_SIMILARITY_ALGORITHM", "cosine")

    # Count files for reporting
    html_files = list(html_dir.glob("*.html"))
    file_count = len(html_files)
    logger.info(get_message("build_index.found_files",
                            file_count=file_count,
                            html_dir=html_dir,
                            limit=limit if limit else "all"))

    # Refresh token before embedding
    refresh_openai_client()

    # Build the index
    return build_index_from_html(
        html_dir=html_dir,
        out_dir=output_dir,
        chunk_strategy=chunk_strategy,
        limit=limit,
        batch_size=batch_size,
        similarity_algorithm=similarity_algorithm,
    )

