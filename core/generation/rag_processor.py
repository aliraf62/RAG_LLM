"""RAG processing for document retrieval and response generation.

Provides utilities for building context from retrieved documents, creating
a formatted prompt, generating a response with an LLM, and adding citations.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol, Callable

from core.rag.context_formatter import build_context_prompt, get_citation_text
from core.rag.conversation import chat_completion
from core.utils.exceptions import RetrievalError
from core.utils.i18n import get_message
from core.rag.prompt_manager import prompt_manager
from core.retrieval.retrieval import get_domain_specific_index, retrieve_documents_with_retry
from core.config.settings import settings
from core.pipeline.retrievers.base import RetrievalResult
from core.rag.classify import classify_question

logger = logging.getLogger(__name__)


class RAGCallback(Protocol):
    """Protocol for RAG callbacks during retrieval and generation."""

    def on_retrieval_start(self) -> None:
        """Called when the retrieval process starts."""
        ...

    def on_chunk_retrieved(self, chunk: RetrievalResult) -> None:
        """Called when a chunk is retrieved from the knowledge base."""
        ...

    def on_retrieval_complete(self, chunks: List[RetrievalResult]) -> None:
        """Called when all chunks have been retrieved."""
        ...

    def on_generation_start(self) -> None:
        """Called when the response generation starts."""
        ...

    def on_token_generated(self, token: str) -> None:
        """Called for each token generated in the response."""
        ...

    def on_generation_complete(self, full_response: str) -> None:
        """Called when response generation is complete."""
        ...


class RAGRequest:
    """Request data for RAG processing.

    Encapsulates query and configuration parameters for document retrieval and LLM completion.
    """

    def __init__(
        self,
        question: str,
        primary_domain: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        user_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
        include_images: bool = True,
        callback: Optional[RAGCallback] = None
    ):
        """Initialize RAG request.

        Args:
            question: User's question text
            primary_domain: Classified domain for the question (e.g., "sourcing", "cso")
            system_prompt: Custom system prompt (defaults to domain-specific from prompt_manager)
            history: Optional conversation history
            model: LLM model to use (defaults to settings.model)
            temperature: Temperature for generation (defaults to settings.default_temperature)
            user_prompt: Custom user prompt template (defaults to standard template)
            top_k: Number of documents to retrieve (defaults to settings.top_k)
            include_images: Whether to include images in the context
            callback: Optional callback for streaming retrieval and generation updates
        """
        self.question = question
        self.primary_domain = primary_domain
        self.system_prompt = system_prompt or prompt_manager.get_system_prompt(primary_domain)
        self.history = history
        self.model = model or settings.model
        self.temperature = temperature or settings.default_temperature
        self.user_prompt = user_prompt
        self.top_k = top_k or settings.top_k
        self.include_images = include_images
        self.callback = callback


class RAGResponse:
    """Response from RAG processing.

    Contains answer text, retrieved documents, citations, and timing information.
    """

    def __init__(
        self,
        answer: str,
        documents: List[Dict[str, any]],
        citations: str = "",
        retrieve_time_ms: int = 0,
        generate_time_ms: int = 0,
        total_time_ms: int = 0,
        error: Optional[str] = None
    ):
        """Initialize RAG response.

        Args:
            answer: Generated answer text
            documents: Retrieved documents used for answering
            citations: Formatted citation text
            retrieve_time_ms: Time spent in document retrieval (ms)
            generate_time_ms: Time spent in LLM generation (ms)
            total_time_ms: Total processing time (ms)
            error: Error message if processing failed
        """
        self.answer = answer
        self.documents = documents
        self.citations = citations
        self.retrieve_time_ms = retrieve_time_ms
        self.generate_time_ms = generate_time_ms
        self.total_time_ms = total_time_ms
        self.error = error

    def get_answer_with_citations(self) -> str:
        """Get answer with citations appended."""
        if not self.citations:
            return self.answer
        return f"{self.answer}\n\n{self.citations}"


def process_rag_query(
    question: str,
    index_dir: str = "",
    history: Optional[List[Dict[str, str]]] = None,
    callback: Optional[RAGCallback] = None,
    **kwargs
) -> Dict[str, Any]:
    """Process a RAG query with streaming updates.

    Args:
        question: The user's question
        index_dir: Optional directory for the vector index
        history: Optional conversation history
        callback: Optional callback for streaming updates
        **kwargs: Additional arguments for processing

    Returns:
        Dict with answer and related information
    """
    start_time = time.time()

    # Classify the question to determine the domain
    domain = classify_question(question)

    # Create a request object
    request = RAGRequest(
        question=question,
        primary_domain=domain,
        history=history,
        callback=callback,
        **kwargs
    )

    # Process the request with streaming updates
    response = process_rag_request(request, index_dir)

    # Calculate total processing time
    total_time = int((time.time() - start_time) * 1000)
    response.total_time_ms = total_time

    return {
        "answer": response.answer,
        "documents": response.documents,
        "citations": response.citations,
        "timing": {
            "retrieve_ms": response.retrieve_time_ms,
            "generate_ms": response.generate_time_ms,
            "total_ms": response.total_time_ms
        }
    }


def process_rag_request(request: RAGRequest, index_dir: str = "") -> RAGResponse:
    """Process a RAG request with streaming updates.

    Args:
        request: The RAG request object
        index_dir: Optional directory for the vector index

    Returns:
        RAGResponse with answer and related information
    """
    callback = request.callback
    documents = []

    # Step 1: Retrieve relevant documents
    retrieve_start = time.time()
    try:
        # Notify callback that retrieval is starting
        if callback:
            callback.on_retrieval_start()

        # Get the domain-specific index
        index_path = get_domain_specific_index(request.primary_domain, index_dir)

        # Convert index_path string to Path object if needed
        from pathlib import Path
        from core.llm import get_embedder

        # Handle both directory structures - with or without a faiss subdirectory
        index_path_obj = Path(index_path)

        # Check if there's a faiss subdirectory and use that if it exists
        faiss_subdir = index_path_obj / "faiss"
        if faiss_subdir.exists() and (faiss_subdir / "index.faiss").exists():
            logger.info(f"Found faiss subdirectory in {index_path_obj}, using it for index")
            index_path_obj = faiss_subdir

        # Check if the index file exists
        index_file = index_path_obj / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found at {index_file}")

        # Look for metadata file
        metadata_path_obj = index_path_obj / "metadata.jsonl"
        if not metadata_path_obj.exists():
            metadata_path_obj = index_path_obj / "metadata.json"
            if not metadata_path_obj.exists():
                logger.warning(f"No metadata file found at {index_path_obj}")
                # Create an empty one as fallback
                metadata_path_obj = index_path_obj / "metadata.jsonl"

        logger.info(f"Using index: {index_file}")
        logger.info(f"Using metadata: {metadata_path_obj}")

        # Get embedder function
        embedder = get_embedder()
        embed_fn = lambda texts: embedder.get_embeddings(texts)

        # Define callback wrapper for retrieve_documents_with_retry
        def retrieval_callback(document):
            # Convert the document to a RetrievalResult
            result = RetrievalResult.from_document(document)
            if callback:
                callback.on_chunk_retrieved(result)

        # Retrieve documents with correct parameter signature
        retrieved_docs = retrieve_documents_with_retry(
            query=request.question,
            embed_fn=embed_fn,
            index_path=index_path_obj,
            metadata_path=metadata_path_obj,
            top_k=request.top_k
        )

        # Convert to standard format
        documents = []
        retrieval_results = []
        for doc in retrieved_docs:
            # Add to documents collection (for the main response)
            documents.append(doc)

            # Create RetrievalResult object directly from document properties
            # instead of using from_document() which expects a langchain Document
            retrieval_results.append(
                RetrievalResult(
                    content=doc.content,
                    metadata=doc.metadata,
                    score=doc.score
                )
            )

        # Notify callback that retrieval is complete
        if callback:
            callback.on_retrieval_complete(retrieval_results)

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        if callback:
            callback.on_retrieval_complete([])
        return RAGResponse(
            answer=get_message("answer.retrieval_error"),
            documents=[],
            error=str(e)
        )

    retrieve_time = int((time.time() - retrieve_start) * 1000)

    # Step 2: Build context prompt from retrieved documents
    context = build_context_prompt(
        question=request.question,
        primary_domain=request.primary_domain,
        docs=documents,
        include_images=request.include_images
    )

    # Step 3: Generate response with LLM
    generate_start = time.time()
    try:
        # Notify callback that generation is starting
        if callback:
            callback.on_generation_start()

        # Define token callback
        def token_callback(token):
            if callback:
                callback.on_token_generated(token)

        # Generate response with streaming
        answer = chat_completion(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt or prompt_manager.get_user_prompt(),
            history=request.history or [],
            context=context,
            question=request.question,
            model=request.model,
            temperature=request.temperature,
            stream=callback is not None,  # Enable streaming if callback is provided
            token_callback=token_callback
        )

        # Notify callback that generation is complete
        if callback:
            callback.on_generation_complete(answer)

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return RAGResponse(
            answer=get_message("answer.generation_error"),
            documents=documents,
            retrieve_time_ms=retrieve_time,
            error=str(e)
        )

    generate_time = int((time.time() - generate_start) * 1000)

    # Step 4: Format citations
    citations = get_citation_text(documents)

    # Return the complete response
    return RAGResponse(
        answer=answer,
        documents=documents,
        citations=citations,
        retrieve_time_ms=retrieve_time,
        generate_time_ms=generate_time
    )


def format_rag_response(response: Dict[str, Any]) -> str:
    """Format a RAG response for display.

    Args:
        response: RAG response dictionary

    Returns:
        str: Formatted answer text with citations
    """
    answer = response["answer"]
    citations = response.get("citations", "")

    if citations:
        return f"{answer}\n\n{citations}"
    return answer
