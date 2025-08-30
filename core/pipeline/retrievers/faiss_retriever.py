from __future__ import annotations
from pathlib import Path
from typing import List, Sequence

from core.utils.component_registry import register
from langchain.schema import Document

from core.pipeline.indexing import FAISSVectorStore
from core.pipeline.retrievers.base import BaseRetriever, RetrievalResult

@register("retriever", "faiss")
class FAISSRetriever(BaseRetriever):
    """
    Retriever implementation using FAISS vector store backend.
    """

    def __init__(self, index_dir: str | Path):
        # Use index.faiss and metadata.jsonl by default
        index_path = Path(index_dir) / "index.faiss"
        meta_path = Path(index_dir) / "metadata.jsonl"
        if not meta_path.exists():
            meta_path = Path(index_dir) / "metadata.json"
        from core.config.settings import settings
        similarity = settings.get("VECTORSTORE_SIMILARITY_ALGORITHM", "cosine")
        self.store = FAISSVectorStore(
            index_path=index_path,
            metadata_path=meta_path,
            similarity=similarity
        )

    def retrieve(self, query: str, top_k: int = 5) -> Sequence[Document]:
        """
        Run a similarity search against the FAISS index.

        Parameters
        ----------
        query : str
            Query text to search for
        top_k : int
            Maximum number of results to return

        Returns
        -------
        Sequence[Document]
            List of relevant documents
        """
        results = self.store.search(query, k=top_k)  # returns List[Document]
        return results

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Run a similarity search against the FAISS index with scores.

        Parameters
        ----------
        query : str
            Query text to search for
        top_k : int
            Maximum number of results to return

        Returns
        -------
        List[RetrievalResult]
            Standardized retrieval results with relevance scores
        """
        # Use similarity_search_with_score if available
        try:
            print(f"DEBUG: Calling similarity_search_with_score with query: '{query}' and top_k: {top_k}")
            docs_and_scores = self.store.similarity_search_with_score(query, k=top_k)
            print(f"DEBUG: Got {len(docs_and_scores) if docs_and_scores else 0} results")

            # Debug the structure of the first result
            if docs_and_scores and len(docs_and_scores) > 0:
                first_item = docs_and_scores[0]
                print(f"DEBUG: First result type: {type(first_item)}, structure: {first_item}")

            results = []

            # Handle the results more carefully to avoid unpacking errors
            for idx, item in enumerate(docs_and_scores):
                print(f"DEBUG: Processing item {idx}, type: {type(item)}, value: {item}")

                # Make sure we only process valid (doc, score) tuples
                if not isinstance(item, tuple):
                    print(f"DEBUG: Item {idx} is not a tuple, skipping")
                    continue

                if len(item) != 2:
                    print(f"DEBUG: Item {idx} doesn't have exactly 2 elements (has {len(item)}), skipping")
                    continue

                try:
                    doc, score = item
                    # Validate the doc object
                    if not hasattr(doc, 'page_content'):
                        print(f"DEBUG: Doc object at {idx} doesn't have page_content attribute")
                        continue

                    # Convert raw similarity to normalized score (higher is better)
                    print(f"DEBUG: Raw score: {score}, type: {type(score)}")
                    normalized_score = 1.0 - (float(score) / 2.0) if float(score) <= 2.0 else 0.0
                    print(f"DEBUG: Normalized score: {normalized_score}")

                    # Create the result object
                    result = RetrievalResult.from_document(doc, score=normalized_score)
                    results.append(result)
                except Exception as item_error:
                    print(f"DEBUG: Error processing item {idx}: {item_error}")
                    import traceback
                    print(traceback.format_exc())
                    continue

            print(f"DEBUG: Returning {len(results)} valid results")
            return results
        except AttributeError as e:
            print(f"DEBUG: AttributeError: {e}, falling back to base implementation")
            # Fallback to base implementation if similarity_search_with_score not available
            return super().retrieve_with_scores(query, top_k)
        except Exception as e:
            # Add better error handling to identify the exact issue
            import traceback
            print(f"ERROR in retrieve_with_scores: {e}")
            print(traceback.format_exc())
            return []  # Return empty list rather than failing completely
