#!/usr/bin/env python3
"""
Vector Store CLI Retrieval Test

This script tests the CLI retrieval functionality with an existing vector store.
It performs queries against the index and outputs the retrieved documents and responses.
"""

import sys
import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import argparse
import time
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import CLI interface components
from cli.chat_cli_interface import StreamingCallback
from core.pipeline.retrievers.base import RetrievalResult
from core.generation.rag_processor import RAGCallback, process_rag_query
from core.pipeline.retrievers.faiss_retriever import FAISSRetriever
from core.rag.classify import classify_question

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class VectorStoreTestCallback(RAGCallback):
    """Callback handler for testing vector store retrieval."""

    def __init__(self, verbose: bool = True):
        """Initialize callback with verbosity setting."""
        self.verbose = verbose
        self.retrieval_started = False
        self.retrieved_chunks = []
        self.retrieval_completed = False
        self.generation_started = False
        self.generated_tokens = []
        self.generation_completed = False
        self.full_response = ""

    def on_retrieval_start(self) -> None:
        """Called when retrieval starts."""
        self.retrieval_started = True
        if self.verbose:
            print("\n[bold yellow]Retrieving relevant information...[/bold yellow]")

    def on_chunk_retrieved(self, chunk: RetrievalResult) -> None:
        """Called when a chunk is retrieved."""
        self.retrieved_chunks.append(chunk)
        if self.verbose:
            source = chunk.metadata.get('source', 'Unknown source')
            score = chunk.score
            print(f"[dim]Found: {source} (score: {score:.4f})[/dim]")

    def on_retrieval_complete(self, chunks: List[RetrievalResult]) -> None:
        """Called when retrieval is complete."""
        self.retrieval_completed = True
        if self.verbose:
            if chunks:
                print(f"\n[green]Retrieved {len(chunks)} relevant passages[/green]")
            else:
                print("\n[yellow]No relevant information found[/yellow]")

    def on_generation_start(self) -> None:
        """Called when generation starts."""
        self.generation_started = True
        if self.verbose:
            print("\n[bold yellow]Generating response...[/bold yellow]\n")

    def on_token_generated(self, token: str) -> None:
        """Called as tokens are generated."""
        self.generated_tokens.append(token)
        self.full_response += token
        if self.verbose:
            # Print tokens in real time
            print(token, end="", flush=True)

    def on_generation_complete(self, full_response: str) -> None:
        """Called when generation is complete."""
        self.generation_completed = True
        self.full_response = full_response
        if self.verbose:
            print("\n\n[green]Generation complete[/green]")


def direct_retrieval_test(index_dir: str, query: str, top_k: int = 5, show_content: bool = False):
    """
    Test direct retrieval from vector store without LLM generation.

    Parameters
    ----------
    index_dir : str
        Path to the vector store directory
    query : str
        Query text to search for
    top_k : int
        Number of results to retrieve
    show_content : bool
        Whether to show the full content of retrieved chunks
    """
    print(f"\n=== Direct Vector Store Retrieval Test ===")
    print(f"Query: \"{query}\"")
    print(f"Vector store: {index_dir}")
    print(f"Retrieving top {top_k} results...\n")

    try:
        # Create retriever
        retriever = FAISSRetriever(index_dir)

        # Time the retrieval
        start_time = time.time()
        results = retriever.retrieve_with_scores(query, top_k=top_k)
        retrieval_time = time.time() - start_time

        # Print results
        print(f"Retrieved {len(results)} results in {retrieval_time:.3f}s\n")

        for i, result in enumerate(results):
            source = result.metadata.get('source', 'Unknown')
            score = result.score

            print(f"\n[Result {i+1}] Score: {score:.4f}")
            print(f"Source: {source}")

            if show_content:
                print(f"Content: {result.content[:300]}...")
                if len(result.content) > 300:
                    print("... (content truncated)")

            # Print other useful metadata if available
            title = result.metadata.get('title')
            if title:
                print(f"Title: {title}")

            # Print any other interesting metadata
            other_keys = [k for k in result.metadata.keys()
                         if k not in ('source', 'title') and not k.startswith('_')]
            if other_keys and show_content:
                print("Other metadata:")
                for key in other_keys:
                    print(f"  - {key}: {result.metadata[key]}")

        return results

    except Exception as e:
        print(f"[bold red]Error in retrieval: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()
        return []


def rag_query_test(index_dir: str, query: str, temperature: float = 0.3,
                  show_citations: bool = True, top_k: int = 5):
    """
    Test full RAG query with retrieval and response generation.

    Parameters
    ----------
    index_dir : str
        Path to the vector store directory
    query : str
        Query text to search for
    temperature : float
        Temperature setting for response generation
    show_citations : bool
        Whether to show citations in the response
    top_k : int
        Number of results to retrieve
    """
    print(f"\n=== Full RAG Query Test ===")
    print(f"Query: \"{query}\"")
    print(f"Vector store: {index_dir}")
    print(f"Temperature: {temperature}")
    print(f"Retrieving top {top_k} results and generating response...\n")

    try:
        # Create callback
        callback = VectorStoreTestCallback(verbose=True)

        # Process RAG query
        start_time = time.time()
        result = process_rag_query(
            question=query,
            index_dir=index_dir,
            callback=callback,
            temperature=temperature,
            top_k=top_k
        )
        processing_time = time.time() - start_time

        # Print statistics
        print(f"\n=== RAG Query Statistics ===")
        print(f"Total processing time: {processing_time:.3f}s")
        print(f"Retrieval time: {result['timing']['retrieve_ms']/1000:.3f}s")
        print(f"Generation time: {result['timing']['generate_ms']/1000:.3f}s")

        # Show citations if requested
        if show_citations and result.get('citations'):
            print(f"\n=== Citations ===\n{result['citations']}")

        return result

    except Exception as e:
        print(f"[bold red]Error in RAG query: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()
        return {}


def conversation_test(index_dir: str, queries: List[str], temperature: float = 0.3, top_k: int = 5):
    """
    Test a multi-turn conversation with history.

    Parameters
    ----------
    index_dir : str
        Path to the vector store directory
    queries : List[str]
        List of queries to ask in sequence
    temperature : float
        Temperature setting for response generation
    top_k : int
        Number of results to retrieve
    """
    print(f"\n=== Multi-turn Conversation Test ===")
    print(f"Vector store: {index_dir}")
    print(f"Temperature: {temperature}")
    print(f"Number of turns: {len(queries)}")

    # Initialize conversation history
    history = []

    for i, query in enumerate(queries):
        print(f"\n[Turn {i+1}] User: {query}\n")

        # Create callback
        callback = VectorStoreTestCallback(verbose=True)

        try:
            # Process RAG query with history
            result = process_rag_query(
                question=query,
                index_dir=index_dir,
                history=history,
                callback=callback,
                temperature=temperature,
                top_k=top_k
            )

            # Update history
            answer = result["answer"]
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})

            # Show history size
            print(f"\nHistory size: {len(history)} messages")

            # Add small delay between turns
            time.sleep(1)

        except Exception as e:
            print(f"[bold red]Error in conversation turn {i+1}: {str(e)}[/bold red]")
            import traceback
            traceback.print_exc()

    return history


def test_advanced_parameters(index_dir: str, query: str, test_params: Dict[str, Any]):
    """
    Test different parameter combinations for RAG.

    Parameters
    ----------
    index_dir : str
        Path to the vector store directory
    query : str
        Query text to search for
    test_params : Dict[str, Any]
        Dictionary of parameter combinations to test
    """
    print(f"\n=== Advanced Parameter Testing ===")
    print(f"Query: \"{query}\"")
    print(f"Vector store: {index_dir}")
    print("Testing parameter combinations:")

    for param_name, values in test_params.items():
        print(f"  - {param_name}: {values}")

    results = {}

    # Generate all parameter combinations
    import itertools
    param_names = list(test_params.keys())
    param_values = list(test_params.values())

    for i, combination in enumerate(itertools.product(*param_values)):
        params = dict(zip(param_names, combination))

        print(f"\n=== Test Combination {i+1} ===")
        for name, value in params.items():
            print(f"{name}: {value}")

        # Create callback
        callback = VectorStoreTestCallback(verbose=True)

        try:
            # Process RAG query with parameters
            start_time = time.time()
            result = process_rag_query(
                question=query,
                index_dir=index_dir,
                callback=callback,
                **params
            )
            processing_time = time.time() - start_time

            # Store result summary
            results[str(params)] = {
                "processing_time": processing_time,
                "retrieval_time": result["timing"]["retrieve_ms"]/1000,
                "generation_time": result["timing"]["generate_ms"]/1000,
                "num_documents": len(result.get("documents", [])),
                "answer_length": len(result.get("answer", ""))
            }

            # Add small delay between tests
            time.sleep(2)

        except Exception as e:
            print(f"[bold red]Error with parameters {params}: {str(e)}[/bold red]")
            import traceback
            traceback.print_exc()

    # Print comparison summary
    print("\n=== Parameter Test Summary ===")
    print(f"{'Parameters':<50} | {'Time (s)':<8} | {'Docs':<5} | {'Ans Len':<7}")
    print("-" * 75)

    for params_str, metrics in results.items():
        print(f"{params_str:<50} | {metrics['processing_time']:<8.3f} | {metrics['num_documents']:<5} | {metrics['answer_length']:<7}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Vector Store CLI Retrieval")
    parser.add_argument("--index-dir", default="/Users/ali.rafieefar/Documents/GitHub/ai_qna_assistant/vector_store/cso_v1/faiss",
                      help="Path to the vector store directory")
    parser.add_argument("--test", choices=["retrieval", "rag", "conversation", "parameters", "all"], default="all",
                      help="Which test to run")
    parser.add_argument("--query", default="How do I create a CSO workflow?",
                      help="Query text to search for")
    parser.add_argument("--top-k", type=int, default=5,
                      help="Number of results to retrieve")
    parser.add_argument("--temperature", type=float, default=0.3,
                      help="Temperature for response generation")
    parser.add_argument("--show-content", action="store_true",
                      help="Show full content of retrieved documents")

    args = parser.parse_args()

    # Check if index exists
    index_path = Path(args.index_dir) / "index.faiss"
    metadata_path = Path(args.index_dir) / "metadata.jsonl"

    if not index_path.exists():
        print(f"Error: index file not found at {index_path}")
        return

    if not metadata_path.exists():
        print(f"Warning: metadata file not found at {metadata_path}")
        metadata_path = Path(args.index_dir) / "metadata.json"
        if not metadata_path.exists():
            print(f"Error: no metadata file found (tried .jsonl and .json)")
            return

    print(f"Using vector store at: {args.index_dir}")
    print(f"Found index file: {index_path}")
    print(f"Found metadata file: {metadata_path}")

    # Run the requested tests
    if args.test in ["retrieval", "all"]:
        direct_retrieval_test(args.index_dir, args.query, args.top_k, args.show_content)

    if args.test in ["rag", "all"]:
        rag_query_test(args.index_dir, args.query, args.temperature, True, args.top_k)

    if args.test in ["conversation", "all"]:
        follow_up_queries = [
            args.query,
            "Can you tell me more details about this?",
            "What are the steps involved?",
            "Are there any best practices I should follow?"
        ]
        conversation_test(args.index_dir, follow_up_queries, args.temperature, args.top_k)

    if args.test in ["parameters", "all"]:
        # Test various parameter combinations
        test_params = {
            "temperature": [0.0, 0.3, 0.7],
            "top_k": [3, 5, 10]
        }
        test_advanced_parameters(args.index_dir, args.query, test_params)


if __name__ == "__main__":
    main()
