"""Higher-level document retrieval for RAG with filtering and domain selection."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import logging
import time
from core.utils.models import RetrievedDocument
from core.config.settings import settings
from core.pipeline.retrievers import create_retriever
from core.utils.exceptions import RetrievalError

logger = logging.getLogger(__name__)

def get_domain_specific_index(primary_domain: str, override_index: str = "") -> str:
    """
    Return the appropriate index path for a domain.

    Args:
        primary_domain: Domain identified from question classification
        override_index: Optional manual override for index location

    Returns:
        Path to the vector index for the specified domain
    """
    logger.debug(f"selecting index for domain: {primary_domain}")
    if override_index:
        return override_index

    vector_stores = settings.get("VECTOR_STORES", {})
    return vector_stores.get(primary_domain, vector_stores.get("default", "vector_store/vector_store_all"))


def retrieve_documents(query: str, embed_fn: Callable[[str], List[float]], index_path: Path, metadata_path: Path, top_k: int = 5) -> List[RetrievedDocument]:
    """
    Retrieve relevant documents from the configured retriever (e.g., FAISS).

    Args:
        query: User query text
        embed_fn: Function to embed the query (not used directly, kept for interface compatibility)
        index_path: Path to FAISS index directory
        metadata_path: Path to document metadata (not used directly, kept for interface compatibility)
        top_k: Number of documents to retrieve

    Returns:
        List of RetrievedDocument objects
    """
    retriever_provider = settings.get("VECTORSTORE_PROVIDER", "faiss")
    # Use create_retriever to instantiate with correct signature
    retriever = create_retriever(retriever_provider, index_dir=str(index_path.parent))
    docs = retriever.retrieve(query, top_k=top_k)
    results = []
    for doc in docs:
        # Try to get score if present (FAISSRetriever may not set it)
        score = getattr(doc, 'score', None)
        if score is None:
            score = doc.metadata.get('score', 0.0) if hasattr(doc, 'metadata') else 0.0
        results.append(RetrievedDocument(
            content=getattr(doc, 'page_content', getattr(doc, 'content', '')),
            score=score,
            metadata=getattr(doc, 'metadata', {}),
            source=getattr(doc, 'metadata', {}).get('source'),
            document_id=getattr(doc, 'metadata', {}).get('document_id'),
        ))
    return results


def retrieve_documents_with_retry(
        query: str,
        embed_fn: Callable[[str], List[float]],
        index_path: Path,
        metadata_path: Path,
        top_k: Optional[int] = None
) -> List[RetrievedDocument]:
    """
    Retrieve relevant documents with error handling and retry logic.

    Args:
        query: User query text
        embed_fn: Function to embed the query
        index_path: Path to FAISS index or directory containing index.faiss
        metadata_path: Path to document metadata
        top_k: Number of documents to retrieve

    Returns:
        List of RetrievedDocument objects

    Raises:
        RetrievalError: If retrieval fails after retries
        IndexError: If index files don't exist
    """
    # Check if index_path points to a directory instead of the actual file
    if index_path.is_dir():
        logger.debug(f"index_path is a directory: {index_path}, looking for index.faiss")
        index_file = index_path / "index.faiss"
    else:
        index_file = index_path

    # Check if metadata_path points to a directory
    if metadata_path.is_dir():
        metadata_file = metadata_path / "metadata.jsonl"
        if not metadata_file.exists():
            metadata_file = metadata_path / "metadata.json"
    else:
        metadata_file = metadata_path

    logger.debug(f"Using index file: {index_file}")
    logger.debug(f"Using metadata file: {metadata_file}")

    # Verify files exist
    if not index_file.exists():
        error_msg = f"Index file not found at {index_file}"
        logger.error(error_msg)
        raise IndexError(error_msg)

    if not metadata_file.exists():
        # We'll still proceed if metadata file is missing
        logger.warning(f"Metadata file not found at {metadata_file}, continuing anyway")

    if top_k is None:
        top_k = settings.top_k

    start_time = time.time()
    logger.debug(f"Retrieving top {top_k} documents for query: '{query[:50]}...'")

    # Create an instance of the retriever directly
    try:
        from core.pipeline.retrievers import create_retriever
        # Create a retriever pointing to the parent directory containing index.faiss
        retriever = create_retriever("faiss", index_dir=str(index_file.parent))

        # Use retrieve_with_scores instead of retrieve to get document-score pairs
        docs_with_scores = retriever.retrieve_with_scores(query, top_k=top_k)

        # Extract the documents directly as RetrievalResult objects don't have a document attribute
        docs = []
        for doc_score in docs_with_scores:
            # Create a RetrievedDocument from the RetrievalResult properties
            if hasattr(doc_score, 'content'):
                # It's likely a RetrievalResult object
                docs.append(RetrievedDocument(
                    content=doc_score.content,
                    score=doc_score.score,
                    metadata=doc_score.metadata,
                    source=doc_score.metadata.get('source'),
                    document_id=doc_score.metadata.get('document_id')
                ))
            elif hasattr(doc_score, 'document'):
                # It might be a tuple or other structure with a document attribute
                doc = doc_score.document
                docs.append(RetrievedDocument(
                    content=getattr(doc, 'page_content', getattr(doc, 'content', '')),
                    score=getattr(doc_score, 'score', 0.0),
                    metadata=getattr(doc, 'metadata', {}),
                    source=getattr(doc, 'metadata', {}).get('source'),
                    document_id=getattr(doc, 'metadata', {}).get('document_id')
                ))
            elif isinstance(doc_score, tuple) and len(doc_score) == 2:
                # It might be a (Document, score) tuple
                doc, score = doc_score
                docs.append(RetrievedDocument(
                    content=getattr(doc, 'page_content', getattr(doc, 'content', '')),
                    score=score,
                    metadata=getattr(doc, 'metadata', {}),
                    source=getattr(doc, 'metadata', {}).get('source'),
                    document_id=getattr(doc, 'metadata', {}).get('document_id')
                ))

        logger.debug(f"Retrieved {len(docs)} documents in {time.time() - start_time:.2f}s")
        return docs

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise RetrievalError(f"Failed to retrieve documents: {e}") from e


def deduplicate_sources(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate documents by source/guide.

    Args:
        docs: List of retrieved documents

    Returns:
        Deduplicated list of documents
    """
    if not settings.get("DEDUPLICATE_SOURCES", True):
        return docs

    seen = set()
    deduplicated = []

    for doc in docs:
        key = doc.get("rating-name") or doc.get("guide-id")
        if key not in seen:
            seen.add(key)
            deduplicated.append(doc)

    return deduplicated

