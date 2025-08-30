#!/usr/bin/env python3
"""
Script to test the CLI streaming functionality.

This script tests the streaming functionality implemented in the CLI,
simulating document retrieval and LLM responses without requiring
an actual vector store or LLM API access.
"""

import sys
import time
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import argparse
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from cli.chat_cli_interface import StreamingCallback
from core.pipeline.retrievers.base import RetrievalResult
from core.generation.rag_processor import RAGCallback

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class MockRetrievalResult(RetrievalResult):
    """Mock implementation of RetrievalResult for testing."""
    pass

class MockRAGCallback(RAGCallback):
    """
    Mock implementation of the RAG callback interface.

    This class records all callback events for verification.
    """
    def __init__(self):
        self.retrieval_started = False
        self.retrieved_chunks = []
        self.retrieval_completed = False
        self.generation_started = False
        self.generated_tokens = []
        self.generation_completed = False
        self.full_response = ""

    def on_retrieval_start(self) -> None:
        self.retrieval_started = True
        logger.info("Retrieval started")

    def on_chunk_retrieved(self, chunk: RetrievalResult) -> None:
        self.retrieved_chunks.append(chunk)
        logger.info(f"Retrieved chunk from: {chunk.metadata.get('source', 'unknown')}")

    def on_retrieval_complete(self, chunks: List[RetrievalResult]) -> None:
        self.retrieval_completed = True
        logger.info(f"Retrieval completed with {len(chunks)} chunks")

    def on_generation_start(self) -> None:
        self.generation_started = True
        logger.info("Generation started")

    def on_token_generated(self, token: str) -> None:
        self.generated_tokens.append(token)
        self.full_response += token
        # Print the token immediately without a newline to simulate streaming
        print(token, end="", flush=True)

    def on_generation_complete(self, full_response: str) -> None:
        self.generation_completed = True
        self.full_response = full_response
        # Print a newline after generation completes
        print()
        logger.info("Generation completed")


class MockStreamingProcessor:
    """
    Mock implementation of a RAG processor for testing the streaming functionality.

    Simulates document retrieval and response generation with configurable delays.
    """
    def __init__(self,
                 retrieval_delay: float = 0.5,
                 generation_delay: float = 0.1,
                 num_chunks: int = 3,
                 response_length: int = 100):
        """
        Initialize the mock processor.

        Parameters
        ----------
        retrieval_delay : float
            Delay between chunk retrievals in seconds
        generation_delay : float
            Delay between token generations in seconds
        num_chunks : int
            Number of chunks to simulate retrieving
        response_length : int
            Length of the simulated response in tokens
        """
        self.retrieval_delay = retrieval_delay
        self.generation_delay = generation_delay
        self.num_chunks = num_chunks
        self.response_length = response_length

        # Mock data
        self.documents = [
            {"title": "Product Overview", "content": "Details about our product features..."},
            {"title": "User Guide", "content": "Step-by-step instructions for users..."},
            {"title": "API Documentation", "content": "Technical details about our API endpoints..."},
            {"title": "FAQ", "content": "Frequently asked questions and answers..."},
            {"title": "Troubleshooting", "content": "Common issues and solutions..."}
        ]

        self.responses = [
            "The product offers several key features that help streamline workflows.",
            "According to the user guide, you can accomplish this in three steps.",
            "Based on the documentation, the API supports both REST and GraphQL interfaces.",
            "The FAQ section addresses this specific question with detailed examples.",
            "I can help you troubleshoot this issue based on the information in our knowledge base."
        ]

    def process_query(self, query: str, callback: Optional[RAGCallback] = None) -> Dict[str, Any]:
        """
        Process a query with simulated retrieval and generation.

        Parameters
        ----------
        query : str
            The user's query
        callback : Optional[RAGCallback]
            Callback for streaming updates

        Returns
        -------
        Dict[str, Any]
            Simulated RAG response
        """
        # Step 1: Simulate retrieval
        if callback:
            callback.on_retrieval_start()

        retrieved_chunks = []
        for i in range(self.num_chunks):
            # Select a random document
            doc_idx = random.randint(0, len(self.documents) - 1)
            doc = self.documents[doc_idx]

            # Create a mock chunk
            chunk = MockRetrievalResult(
                content=doc["content"],
                metadata={
                    "source": f"document_{doc_idx}.md",
                    "title": doc["title"]
                },
                score=random.random()
            )

            # Add to results
            retrieved_chunks.append(chunk)

            # Call callback if provided
            if callback:
                callback.on_chunk_retrieved(chunk)

            # Simulate delay between retrievals
            time.sleep(self.retrieval_delay)

        # Complete retrieval
        if callback:
            callback.on_retrieval_complete(retrieved_chunks)

        # Step 2: Simulate generation
        if callback:
            callback.on_generation_start()

        # Select a random response
        response_idx = random.randint(0, len(self.responses) - 1)
        full_response = self.responses[response_idx]

        # Stream tokens
        if callback:
            # Generate tokens with delay
            for token in full_response.split():
                callback.on_token_generated(token + " ")
                time.sleep(self.generation_delay)

            # Complete generation
            callback.on_generation_complete(full_response)

        # Return response
        return {
            "answer": full_response,
            "documents": [chunk.to_dict() for chunk in retrieved_chunks],
            "citations": "Source: Mock Documentation",
            "timing": {
                "retrieve_ms": int(self.retrieval_delay * self.num_chunks * 1000),
                "generate_ms": int(self.generation_delay * len(full_response.split()) * 1000),
                "total_ms": int((self.retrieval_delay * self.num_chunks +
                                self.generation_delay * len(full_response.split())) * 1000)
            }
        }


def test_cli_callback():
    """Test the StreamingCallback with mock data."""
    print("\n=== Testing CLI StreamingCallback ===\n")

    # Create a callback instance
    callback = StreamingCallback()

    # Create a mock processor
    processor = MockStreamingProcessor(
        retrieval_delay=0.3,  # Faster retrieval for testing
        generation_delay=0.1,  # Faster generation for testing
        num_chunks=3,
        response_length=20
    )

    # Process a query with the callback
    query = "How do I set up the product?"
    print(f"\nUser query: {query}\n")

    # Process the query with the callback
    result = processor.process_query(query, callback)

    # Display the final answer for verification
    print("\n=== Final Answer ===")
    print(result["answer"])
    print("\n=== Citations ===")
    print(result["citations"])

    # Display timing information
    print("\n=== Timing Information ===")
    print(f"Retrieval time: {result['timing']['retrieve_ms']} ms")
    print(f"Generation time: {result['timing']['generate_ms']} ms")
    print(f"Total time: {result['timing']['total_ms']} ms")

    return callback


def demo_cli_interface():
    """
    Simulate the CLI interface with mock data.

    This function creates a simple interactive demo that mimics
    the behavior of the actual CLI interface but uses mock data.
    """
    print("\n=== CLI Interactive Demo (Mock Mode) ===\n")
    print("This demo simulates the CLI interface with mock data.")
    print("Type '/exit' to quit.\n")

    # Create a mock processor
    processor = MockStreamingProcessor(
        retrieval_delay=0.5,
        generation_delay=0.1,
        num_chunks=3,
        response_length=50
    )

    while True:
        # Get user input
        user_input = input("\n\033[1;34mYou: \033[0m")

        # Check for exit command
        if user_input.lower() in ['/exit', '/quit']:
            break

        # Process the query
        callback = StreamingCallback()
        print("\033[1;32mAssistant: \033[0m")
        result = processor.process_query(user_input, callback)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test CLI streaming functionality")
    parser.add_argument("--mode", choices=["test", "demo"], default="demo",
                      help="Test mode: 'test' for automated test, 'demo' for interactive demo")

    args = parser.parse_args()

    if args.mode == "test":
        test_cli_callback()
    else:
        demo_cli_interface()


if __name__ == "__main__":
    main()
