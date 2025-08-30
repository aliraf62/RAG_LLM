"""
Abstract base for retriever components.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence, Dict, Any, Optional, List
from core.config.settings import settings
from langchain.schema import Document


@dataclass
class RetrievalResult:
    """
    Standardized representation of a document retrieved from a knowledge base.

    Provides a consistent interface across different retrieval backends,
    decoupling downstream components from specific retriever implementations.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @classmethod
    def from_document(cls, doc: Document, score: Optional[float] = None) -> 'RetrievalResult':
        """
        Convert a Langchain Document to a RetrievalResult.

        Parameters
        ----------
        doc : Document
            Langchain Document object
        score : float, optional
            Relevance score (if available)

        Returns
        -------
        RetrievalResult
            Standardized retrieval result
        """
        return cls(
            content=doc.page_content,
            metadata=doc.metadata,
            score=score or 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the retrieval result
        """
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score
        }


class BaseRetriever(ABC):
    """Provider-agnostic document retriever."""

    CATEGORY = "retriever"

    @abstractmethod
    def retrieve(self, query: str, top_k: int = settings.get("TOP_K", 5)) -> Sequence[Document]:
        """Return *top_k* documents relevant to *query*."""

    def retrieve_with_scores(self, query: str, top_k: int = settings.get("TOP_K", 5)) -> List[RetrievalResult]:
        """
        Retrieve documents with relevance scores.

        Default implementation calls retrieve() and converts Documents to RetrievalResults.
        Subclasses should override this if they can provide native scoring.

        Parameters
        ----------
        query : str
            Query text
        top_k : int
            Maximum number of results to return

        Returns
        -------
        List[RetrievalResult]
            Standardized retrieval results with relevance scores
        """
        docs = self.retrieve(query, top_k)
        return [RetrievalResult.from_document(doc) for doc in docs]

