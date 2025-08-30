"""Entry‑point for the Coupa AIQA Assistant."""
from __future__ import annotations

import typer
import logging
from pathlib import Path

from core.config.settings import settings
from core.services.customer_service import customer_service
from core.llm.providers.openai import
from core.pipeline.indexing.indexer import build_index_from_documents
from cli.chat_cli_interface import run_chat_interface
from cli.commands import app as cli_app
from cli.auth_commands import app as auth_app
from core.auth import get_current_context, restore_session_from_file

# Create main Typer app with sub-commands
app = typer.Typer(help="Coupa AIQA Assistant with RAG capabilities.")

# Add sub-commands
app.add_typer(cli_app, name="cli", help="Original CLI commands")
app.add_typer(auth_app, name="auth", help="Authentication commands")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_and_build_index() -> None:
    """Check if vector index exists and build if needed."""
    vector_store_path = Path(settings.get("VECTOR_STORE_PATH"))
    index_path = vector_store_path / "index.faiss"

    if not index_path.exists():
        typer.echo("No document index found. Building index from parquet files...")
        build_index_from_documents()
        typer.echo(f"Index built successfully at {index_path}")


@app.command("chat")
def chat_command(
        index_dir: str = typer.Option(
            "",
            help="Custom vector store directory (defaults to config.VECTOR_STORE_PATH)"
        ),
        top_k: int = typer.Option(
            None,
            help="Number of documents to retrieve (defaults to config.TOP_K)"
        ),
        customer: str = typer.Option(
            None,
            help="Specify the customer to load data for"
        )
):
    """Launch interactive chat interface."""
    # Initialize client and services
    client = refresh_openai_client()

    # Load customer data if specified
    if customer:
        typer.echo(f"Loading data for customer: {customer}")
        customer_service.load_customer(customer)

    # Check index exists
    check_and_build_index()

    # Create embedding function with batch handling
    def embedding_function(text):
        if isinstance(text, list):
            return [get_embeddings(t, client) for t in text]
        return get_embeddings(text, client)

    # Run the interactive interface
    run_chat_interface()


@app.command("cli")
def cli_command():
    """Launch Typer CLI with slash commands."""
    cli_app()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Main entry point that runs chat by default."""
    restore_session_from_file()
    if ctx.invoked_subcommand is None:
        chat_command()


if __name__ == "__main__":
    app()
