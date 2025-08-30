# core/models.py
"""Domain models for the RAG system.

Defines typed data classes for documents, chunks, search results, and other
core data structures used throughout the system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None


@dataclass
class CleanResult:
    """Result of a document cleaning operation."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_success: bool = True
    error_message: Optional[str] = None

@dataclass
class CleaningOptions:
    """Configuration for document cleaning operations."""
    normalize_whitespace: bool = True
    normalize_newlines: bool = True
    extract_metadata: bool = True
    output_format: str = "text"
    max_length: Optional[int] = None


@dataclass
class RetrievedDocument:
    """A document retrieved from the vector store."""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    document_id: Optional[str] = None

@dataclass
class Row:
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured: Dict[str, Any] = field(default_factory=dict)
    assets: List[str] = field(default_factory=list)

@dataclass
class RAGRequest:
    """Request parameters for a RAG query."""
    question: str
    index_dir: Optional[Union[str, Path]] = None
    raw_output: bool = False
    top_k: Optional[int] = None


@dataclass
class RAGResponse:
    """Response from a RAG query."""
    question: str
    classification: Dict[str, Any]
    index_path: str
    docs: List[RetrievedDocument]
    docs_count: int
    raw_output: bool
    answer: Optional[str] = None


@dataclass
class IndexBuildResult:
    """Result of an index building operation."""
    index_path: Path
    document_count: int
    chunk_count: int
    is_success: bool = True
    error_message: Optional[str] = None


@dataclass
class ExportResult:
    """Result of an export operation."""
    output_path: Path
    document_count: int
    is_success: bool = True
    error_message: Optional[str] = None

@dataclass
class PromptTemplate:
    """Template for system and user prompts."""
    system_prompt: str
    user_prompt: Optional[str] = None

@dataclass
class EmbeddingResult:
    """Result of document embedding operation."""
    vector: List[float]
    document_id: str
    is_success: bool = True
    error_message: Optional[str] = None

@dataclass
class RetrievalOptions:
    """Options for document retrieval."""
    top_k: int
    similarity_threshold: float = 0.0
    filter_low_scores: bool = False

@dataclass
class ChunkingOptions:
    """Configuration for document chunking."""
    chunk_size: int
    chunk_overlap: int
    chunk_strategy: str = "header"

@dataclass
class ChunkResult:
    """Result of document chunking operations."""
    chunks: List[DocumentChunk]
    total_chunks: int
    is_success: bool = True
    error_message: Optional[str] = None

@dataclass
class ExporterArgs:
    """Arguments for document exporters."""
    out_dir: Path
    limit: Optional[int] = None
    no_captions: bool = False
    force_regenerate: bool = False
    workbook: Optional[Path] = None
    assets_dir: Optional[Path] = None
    parquet_file: Optional[Path] = None


@dataclass
class DocumentMetadata:
    """Standard metadata extracted from documents."""
    title: Optional[str] = None
    source: Optional[str] = None
    content_type: Optional[str] = None
    document_purpose: Optional[str] = None
    has_images: bool = False
    has_tables: bool = False
    has_video: bool = False
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {k: v for k, v in asdict(self).items() if v is not None}
        result.update(self.custom_fields)
        return result


class SingletonMeta(type):
    """
    Metaclass that implements the Singleton pattern.

    This metaclass ensures that only one instance of a class is created.
    All subsequent instantiations return the same instance.

    Example
    -------
    >>> class MyClass(metaclass=SingletonMeta):
    ...     pass
    >>> a = MyClass()
    >>> b = MyClass()
    >>> a is b  # True
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Return existing instance if it exists, otherwise create new instance."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

