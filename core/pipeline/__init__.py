"""
Data ingestion package for document processing, indexing, and retrieval.

Provides a complete pipeline for ingesting documents from various sources,
cleaning and chunking them, embedding into vector stores, and retrieving
relevant documents for RAG applications.
"""

# Base classes
from core.pipeline.base import Row, BasePipelineComponent

# Chunkers
from core.pipeline.chunkers.html_chunker import HTMLChunker
from core.pipeline.chunkers.markdown_chunker import MarkdownChunker
from core.pipeline.chunkers.text_chunker import TextChunker

# Cleaners
from core.pipeline.cleaners.html_cleaner import HTMLCleaner
from core.pipeline.cleaners.markdown_cleaner import MarkdownCleaner
from core.pipeline.cleaners.text_cleaner import TextCleaner

# Loaders
from core.pipeline.loaders.html_fs_loader import HTMLFSLoader
from core.pipeline.loaders.parquet_document_loader import ParquetDocumentLoader

# Embedders


# Exporters
from core.pipeline.exporters.parquet_html_exporter import ParquetHTMLExporter
from core.pipeline.exporters.txt_to_html_exporter import TxtToHTMLExporter
from core.pipeline.exporters.factory import create_exporter

# Indexing
from core.pipeline.indexing.base import BaseVectorStore
from core.pipeline.indexing.faiss import FAISSVectorStore
from core.pipeline.indexing.indexer import build_index_from_documents

# Retrievers
from core.pipeline.retrievers import get_retriever, list_available_retrievers
from core.pipeline.embedders.openai_embedder import OpenAIEmbedder

__all__ = [
    # Base classes
    "Row",
    "BasePipelineComponent",

    # Chunkers
    "HTMLChunker",
    "MarkdownChunker",
    "TextChunker",

    # Cleaners
    "HTMLCleaner",
    "MarkdownCleaner",
    "TextCleaner",

    # Loaders
    "HTMLFSLoader",
    "ParquetDocumentLoader",

    # Embedders
    "OpenAIEmbedder",

    # Exporters
    "ParquetHTMLExporter",
    "TxtToHTMLExporter",
    "create_exporter",

    # Extractors

    # Indexing
    "BaseVectorStore",
    "FAISSVectorStore",
    "build_index_from_documents",

    # Retrievers
    "get_retriever",
    "list_available_retrievers",
]

