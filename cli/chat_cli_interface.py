"""Interactive chat interface for Coupa AI Assistant.

Provides a terminal-based chat interface with RAG (Retrieval-Augmented Generation) capabilities.
Supports slash commands, knowledge base queries, and interactive Q&A.
See docs/architecture/chat_cli_interface.md for detailed design and usage.
"""

from __future__ import annotations

from typing import List

from rich import print
from rich.markdown import Markdown
from rich.prompt import Prompt

from core.generation.rag_processor import process_rag_query, format_rag_response
from core.pipeline.retrievers.base import RetrievalResult
from .commands import dispatch


class StreamingCallback:
    """
    Callback handler for streaming retrieval and generation results to the terminal.

    Provides real-time feedback on the retrieval and generation process,
    making the experience more interactive and engaging.
    """
    def __init__(self):
        self.retrieved_chunks = []
        self.current_generation = ""
        self.is_retrieving = False
        self.is_generating = False

    def on_retrieval_start(self) -> None:
        """Called when the retrieval process starts."""
        self.is_retrieving = True
        print("[bold yellow]Retrieving relevant information...[/bold yellow]")

    def on_chunk_retrieved(self, chunk: RetrievalResult) -> None:
        """Called when a chunk is retrieved from the knowledge base."""
        self.retrieved_chunks.append(chunk)
        source = chunk.metadata.get('source', 'Unknown source')
        print(f"[dim]Found relevant information in: {source}[/dim]")

    def on_retrieval_complete(self, chunks: List[RetrievalResult]) -> None:
        """Called when all chunks have been retrieved."""
        self.is_retrieving = False
        if chunks:
            print(f"[green]Retrieved {len(chunks)} relevant passages[/green]")
        else:
            print("[yellow]No specific information found in the knowledge base[/yellow]")

    def on_generation_start(self) -> None:
        """Called when the response generation starts."""
        self.is_generating = True
        print("[bold yellow]Generating response...[/bold yellow]")

    def on_token_generated(self, token: str) -> None:
        """Called for each token generated in the response."""
        self.current_generation += token
        # Print the token in real-time
        print(token, end="", flush=True)

    def on_generation_complete(self, full_response: str) -> None:
        """Called when response generation is complete."""
        self.is_generating = False
        self.current_generation = full_response


def run_chat_interface():
    """Run interactive chat interface with slash command support.

    Launches a terminal-based chat loop, supporting both natural language
    questions and slash commands for knowledge base operations.

    Returns:
        None
    """
    print("[bold green]Coupa AI Assistant[/bold green] - Type '/help' for commands or '/exit' to quit")

    history = []

    while True:
        user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

        if user_input.lower() in ('/exit', '/quit'):
            break

        if user_input.startswith('/') and user_input != '/help':
            result = dispatch(user_input)
            if result:
                print(f"[bold green]Assistant[/bold green]\n{Markdown(result)}")
            continue

        if user_input == '/help':
            print("[bold green]Assistant[/bold green]\nAvailable commands:")
            print("  /query [--raw] [--k NUM] QUESTION - Search knowledge base")
            print("  /build-index <html_dir> <out_dir> - Build index from HTML files")
            print("  /ping - Test if system is responsive")
            print("  /exit - Exit the chat")
            continue

        try:
            # Initialize streaming callback
            streaming_callback = StreamingCallback()

            # Process the RAG query with streaming callback
            result = process_rag_query(
                question=user_input,
                index_dir="",
                history=history,
                callback=streaming_callback
            )

            # Format and display the final response
            response = format_rag_response(result)
            print(f"[bold green]Assistant[/bold green]\n{Markdown(response)}")

            # Update conversation history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    run_chat_interface()

