# pipeline/loaders/parquet_document_loader.py
"""
Load Coupa Parquet documents lazily and process them into chunks.
"""
from __future__ import annotations

import logging
import os
import pathlib
from typing import Iterable, List, Dict, Any, Optional, Iterator, Tuple

import pandas as pd
import pyarrow.dataset as ds

from core.config.settings import settings
from core.utils.i18n import get_message
from core.utils.component_registry import component_message
from core.pipeline.chunkers.text_chunker import TextChunker
from core.pipeline.cleaners.html_cleaner import HTMLCleaner
from .base import BaseLoader
from core.utils.component_registry import register

logger = logging.getLogger(__name__)
__all__ = ["ParquetDocumentLoader", "discover_parquet_files"]


def discover_parquet_files(data_dir: str = None) -> List[str]:
    """
    Automatically discover all parquet files in the data directory.

    Args:
        data_dir: Directory to search for parquet files

    Returns:
        List of paths to parquet files
    """
    if data_dir is None:
        data_dir = settings.get("DATASETS_PATH", "data/datasets")

    parquet_files = []
    try:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
    except Exception as e:
        logger.error(get_message("loader.parquet.error_discovery", error=str(e)))
    return parquet_files

@register("loader", "parquet")
class ParquetDocumentLoader(BaseLoader):
    """Stream Parquet rows without loading the entire file in memory."""

    REQUIRED_COLUMNS = {"link", "content"}

    def __init__(self, parquet_path: str | pathlib.Path, **kwargs):
        """
        Initialize the Parquet loader.

        Args:
            parquet_path: Path to parquet file

        Raises:
            FileNotFoundError: If parquet file not found
        """
        super().__init__(**kwargs)
        self.dataframe = None
        self.path = pathlib.Path(parquet_path)
        if not self.path.exists():
            logger.error(get_message("loader.parquet.file_not_found", path=str(self.path)))
            raise FileNotFoundError(get_message("loader.parquet.file_not_found", path=str(self.path)))

        try:
            self._dataset = ds.dataset(self.path)
            # Create instances of cleaners and chunkers
            self._html_cleaner = HTMLCleaner()
            self._chunker = TextChunker()
            # Validate schema
            self._validate_schema()
        except Exception as e:
            logger.error(f"Error initializing ParquetDocumentLoader: {e}")
            raise ValueError(get_message("loader.parquet.error_init", error=str(e)))

    def load_all_parquets(self, data_dir: str = None) -> Optional[pd.DataFrame]:
        """
        Load all parquet files in the data directory.

        Args:
            data_dir: Directory containing parquet files

        Returns:
            Pandas DataFrame with combined data

        Raises:
            ValueError: If no parquet files found
        """
        parquet_files = discover_parquet_files(data_dir)
        if not parquet_files:
            logger.error(f"No parquet files found in {data_dir}")
            raise ValueError(get_message("loader.parquet.no_files", directory=data_dir))

        dfs = []
        for file_path in parquet_files:
            df = self.load_parquet(file_path)
            if df is not None:
                dfs.append(df)

        if dfs:
            self.dataframe = pd.concat(dfs, ignore_index=True)
            return self.dataframe
        return None

    def _iter_sources(self) -> Iterator[Tuple[pathlib.Path, str, Dict[str, Any], Dict[str, Any], List[str]]]:
        """
        Implement BaseLoader's _iter_sources method with error handling.

        Returns:
            Iterator of tuples: (path, text, metadata, structured_data, assets)
        """
        batch_size = settings.get("PARQUET_BATCH_SIZE", 1024)
        for row in self.iter_rows(batch_size):
            try:
                # Create a synthetic path based on the row ID or other unique identifier
                doc_id = row.get("id", f"doc_{id(row)}")
                path = self.path.with_name(f"{self.path.stem}_{doc_id}.html")

                # Extract metadata from the row
                metadata = {
                    "doc_id": doc_id,
                    "link": row.get("link", "unknown_link"),
                    "source": row.get("link", "unknown_link"),
                    "format": "html_from_parquet",
                }

                # Add any additional row metadata
                for k, v in row.items():
                    if k != "content" and isinstance(v, (str, int, float, bool)):
                        metadata[k] = v

                # Create structured data - include the full row as structured data
                structured_data = {"row": {k: v for k, v in row.items() if k != "content"}}

                # Extract any assets if they exist in the row
                assets = []
                if "assets" in row and isinstance(row["assets"], list):
                    assets = row["assets"]
                elif "images" in row and isinstance(row["images"], list):
                    assets = row["images"]

                yield path, row["content"], metadata, structured_data, assets

            except Exception as e:
                logger.error(get_message("loader.error.processing_row", id=row.get("id", "unknown"), error=str(e)))
                # Yield a minimal valid result to avoid breaking the pipeline
                path = self.path.with_name(f"{self.path.stem}_{row.get('id', 'unknown')}.html")
                yield path, f"Error processing row: {e}", {"error": str(e), "doc_id": row.get("id", f"doc_{id(row)}")}, {}, []

    def iter_rows(self, batch_size: int = None) -> Iterable[Dict[str, Any]]:
        """
        Iterate through rows in the parquet file in batches with error handling.
        """
        if batch_size is None:
            batch_size = settings.get("PARQUET_BATCH_SIZE", 1024)

        try:
            for batch in self._dataset.to_batches(batch_size=batch_size):
                try:
                    tbl = batch.to_pydict()
                    for idx in range(len(batch)):
                        try:
                            yield {col: tbl[col][idx] for col in tbl}
                        except Exception as e:
                            logger.error(f"Error processing row at index {idx}: {e}")
                            # Skip this row and continue
                            continue
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # Continue to next batch
                    continue
        except Exception as e:
            logger.error(f"Error iterating through dataset: {e}")
            # Yield an empty row to avoid breaking the pipeline completely
            yield {"error": str(e), "content": f"Error reading parquet file: {e}", "link": str(self.path)}

    def _validate_schema(self) -> None:
        """
        Validate that required columns exist.

        Raises:
            ValueError: If required columns are missing
        """
        try:
            cols = set(self._dataset.schema.names)
            if not self.REQUIRED_COLUMNS.issubset(cols):
                logger.error(f"Parquet schema missing required columns: {self.REQUIRED_COLUMNS}")
                raise ValueError(get_message("loader.parquet.schema_missing_columns",
                                             required_columns=self.REQUIRED_COLUMNS))
        except Exception as e:
            logger.error(f"Error validating schema: {e}")
            raise ValueError(get_message("loader.parquet.error_validating", error=str(e)))

    @staticmethod
    def load_parquet(file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a single parquet file.

        Args:
            file_path: Path to parquet file

        Returns:
            DataFrame or None if loading failed
        """
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Error loading parquet file {file_path}: {e}")
            return None

    def to_chunks(
        self,
        max_tokens: int = None,
        overlap: int = None,
        batch_size: int = None,
        rag_options: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return *all* chunks (eager).

        For gigantic corpora use `iter_chunks` instead.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks
            batch_size: Batch size for processing
            rag_options: Options for RAG processing

        Returns:
            List of chunked documents
        """
        try:
            return list(self.iter_chunks(max_tokens, overlap, batch_size, rag_options))
        except Exception as e:
            logger.error(f"Error converting dataset to chunks: {e}")
            return []

    def iter_chunks(
            self,
            max_tokens: int = None,
            overlap: int = None,
            batch_size: int = None,
            rag_options: Dict[str, Any] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Lazily yield cleaned + chunked documents with error handling.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks
            batch_size: Batch size for processing
            rag_options: Options for RAG processing

        Returns:
            Iterator of chunked documents
        """
        # Get default values from configuration
        if max_tokens is None:
            max_tokens = settings.get("DEFAULT_CHUNK_SIZE", 800)
        if overlap is None:
            overlap = settings.get("DEFAULT_CHUNK_OVERLAP", 100)
        if batch_size is None:
            batch_size = settings.get("PARQUET_BATCH_SIZE", 256)

        # Default RAG options
        if rag_options is None:
            rag_options = settings.get("PARQUET_LOADER_OPTIONS", {})
            if not rag_options:  # Fallback if not configured
                rag_options = {
                    "preserve_tables": settings.get("HTML_CLEANER_PRESERVE_TABLES", True),
                    "extract_metadata": settings.get("HTML_CLEANER_EXTRACT_METADATA", True),
                    "preserve_heading_hierarchy": settings.get("HTML_CLEANER_PRESERVE_HEADING_HIERARCHY", True),
                }

        for i, row in enumerate(self.iter_rows(batch_size)):
            try:
                # Use the HTML cleaner's RAG-specific method
                clean_result = self._html_cleaner.clean_for_rag(row["content"], **rag_options)

                # Extract the cleaned text and any extracted metadata
                clean_text = clean_result["text"]
                html_metadata = clean_result.get("metadata", {})

                doc_id = row.get("id", f"doc_{i}")

                # Create metadata dictionary, combining HTML metadata and document info
                metadata = {
                    "doc_id": doc_id,
                    "url": row.get("link", ""),
                    **html_metadata  # Add any extracted HTML metadata (title, etc.)
                }

                # Use the chunker instance
                documents = self._chunker.chunk(
                    content=clean_text,
                    metadata=metadata,
                    chunk_size=max_tokens,
                    chunk_overlap=overlap
                )

                # Convert documents back to the expected dictionary format
                for doc in documents:
                    try:
                        yield {
                            "doc_id": doc.metadata.get("doc_id", doc_id),
                            "chunk_id": doc.metadata.get("chunk_index", 0),
                            "text": doc.page_content,
                            "url": doc.metadata.get("url", row.get("link", "")),
                            "token_count": doc.metadata.get("token_count", 0),
                            "start_word_idx": doc.metadata.get("start_word_idx", 0),
                            "end_word_idx": doc.metadata.get("end_word_idx", 0),
                            "title": doc.metadata.get("title", ""),
                        }
                    except Exception as e:
                        logger.error(f"Error yielding chunk for document {doc_id}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error processing row {i} with ID {row.get('id', 'unknown')}: {e}")
                # Skip this row and continue to the next one
                continue

