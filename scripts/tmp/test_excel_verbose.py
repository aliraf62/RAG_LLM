#!/usr/bin/env python
"""
Verbose Excel testing script with enhanced visualization for AI QnA Assistant.

This script is similar to test_excel_end_to_end.py but with detailed logging and
visualization of each processing stage. It allows you to see exactly what's happening
during extraction, indexing, and question answering.

Example usage:
python scripts/test_excel_verbose.py --excel "customers/coupa/datasets/in-product-guides-Guide+Export/Workflow Steps.xlsb" --customer coupa --extractor cso_workflow --limit 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from rich.panel import Panel

# Add parent directory to path to import from the root package
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils.component_registry import create_component_instance
from core.config.settings import settings
from core.llm import get_llm_client
from core.pipeline.indexing.indexer import build_index_from_documents
from core.generation.rag_processor import process_rag_request, RAGRequest
from core.utils.component_registry import get as get_component
from core.services.customer_service import customer_service
from langchain.schema import Document

# Import our enhanced logging utilities
from scripts.enhanced_logging import (
    console, log_stage, show_document_sample,
    create_progress_bar, timed_operation
)

# Ensure customer package is imported to register components
import customers.coupa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@log_stage("Excel ETL Pipeline")
def execute_excel_etl(
    excel_file_str: str,
    customer: str,
    extractor: str,
    limit: int,
    dataset_id_str: str
):
    """Execute the full ETL pipeline with enhanced visualization."""
    excel_path = Path(excel_file_str)
    if not excel_path.exists():
        console.print(f"[bold red]Excel file not found: {excel_path}[/bold red]")
        return None

    console.print(Panel(f"[bold green]Starting ETL for[/bold green] [cyan]{excel_path.name}[/cyan]\n"
                        f"[bold]Customer:[/bold] {customer}\n"
                        f"[bold]Extractor:[/bold] {extractor}\n"
                        f"[bold]Document Limit:[/bold] {limit}\n"
                        f"[bold]Dataset ID:[/bold] {dataset_id_str}"))

    # Configure settings with useful defaults for testing
    configure_test_settings(customer, excel_path, limit)

    # Extract data from the Excel file
    documents = extract_data_with_details(customer, excel_path, extractor, limit)
    if not documents:
        return None

    # Build the FAISS index from documents
    vector_store_path = build_index_with_progress(customer, documents, dataset_id_str)
    if not vector_store_path:
        return None

    # Return the results
    return {
        "documents": documents,
        "vector_store_path": vector_store_path,
        "document_count": len(documents),
        "excel_path": excel_path
    }

@timed_operation
def configure_test_settings(customer_id: str, excel_path: Path, limit: int = 5):
    """Configure settings for the test run with visual feedback"""
    console.print("[bold blue]Configuring test settings...[/bold blue]")

    # Load customer configuration
    customer_service.load_customer(customer_id)

    # Define overrides with keys matching AppConfig field names (assumed lowercase)
    field_overrides = {
        "export_limit": limit,
        "test_excel_path": str(excel_path),
        "enable_citations": True
    }

    # Create a table to display settings
    from rich.table import Table
    table = Table(title="Test Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for key, value in field_overrides.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
            table.add_row(key, str(value))
        else:
            table.add_row(key, f"[red]Not available[/red]")

    console.print(table)
    return True

@log_stage("Data Extraction")
def extract_data_with_details(customer_id: str, excel_path: Path, extractor_name: str, limit: int = 5):
    """Extract data from Excel with detailed visualization of the process and results"""
    extractor_class = get_component("extractor", extractor_name)

    console.print(f"[bold]Using extractor:[/bold] {extractor_name} ({extractor_class.__name__})")

    # Instantiate the extractor
    extractor = create_component_instance(
        "extractor",
        extractor_name,
        file_path=excel_path,
        customer_id=customer_id
    )

    # Extract rows with progress indication
    console.print("[bold]Extracting rows from Excel...[/bold]")
    rows = []

    # Create progress tracking
    with console.status("[bold green]Extracting data from Excel...") as status:
        for i, row_data in enumerate(extractor.extract_rows()):
            rows.append(row_data)
            if i % 10 == 0:  # Update status periodically
                status.update(f"[bold green]Extracting row {i+1}...")
            if i + 1 >= limit:
                break

    console.print(f"[green]Extracted {len(rows)} rows from Excel[/green]")

    # Show sample of raw extracted data
    if rows:
        from rich.pretty import Pretty
        console.print("[bold]Sample of Raw Extracted Data:[/bold]")
        sample_row = rows[0]
        console.print(Pretty(sample_row))

    # Convert to Langchain Documents with progress
    documents = []
    console.print("[bold]Converting to Langchain Documents...[/bold]")

    for i, row_dict in enumerate(rows):
        try:
            if 'text' not in row_dict or 'metadata' not in row_dict:
                console.print(f"[yellow]Row {i} missing 'text' or 'metadata' key - skipping[/yellow]")
                continue

            doc = Document(
                page_content=str(row_dict["text"]),
                metadata=row_dict["metadata"]
            )
            documents.append(doc)
        except Exception as e:
            console.print(f"[red]Error converting row {i}: {str(e)}[/red]")

    console.print(f"[green]Converted {len(documents)} rows to Langchain Documents[/green]")

    # Show sample documents
    show_document_sample(documents)

    return documents

@log_stage("Index Building")
def build_index_with_progress(customer_id: str, documents: list, dataset_id: str = "excel_verbose_test"):
    """Build the FAISS index with progress visualization"""
    if not documents:
        console.print("[bold red]No documents to index[/bold red]")
        return None

    console.print(f"[bold]Building index for {len(documents)} documents...[/bold]")
    console.print(f"[bold]Dataset ID:[/bold] {dataset_id}")

    with console.status("[bold green]Building vector index...") as status:
        # Build index with progress visualization (implementation uses its own progress)
        vector_store_path = build_index_from_documents(
            docs=documents,
            customer_id=customer_id,
            dataset_id=dataset_id
        )

    if vector_store_path:
        console.print(f"[green]Index built successfully at:[/green] {vector_store_path}")

        # Show the index directory contents
        index_dir = Path(vector_store_path)
        if index_dir.exists():
            files = list(index_dir.glob("*"))

            from rich.table import Table
            table = Table(title="Vector Store Files")
            table.add_column("Filename", style="cyan")
            table.add_column("Size", style="green")

            for file in files:
                size = file.stat().st_size / 1024  # KB
                size_str = f"{size:.2f} KB"
                table.add_row(file.name, size_str)

            console.print(table)
    else:
        console.print("[bold red]Failed to build index[/bold red]")

    return vector_store_path

@log_stage("Question Answering")
def test_question_answering(vector_store_path: str, question: str):
    """Test the question answering with detailed output"""
    console.print(f"[bold]Testing question:[/bold] \"{question}\"")

    # Initialize OpenAI client
    client = get_llm_client()
    console.print(f"[bold]Using LLM provider:[/bold] {settings.get('LLM_PROVIDER', 'openai')}")

    # Initialize vector store
    from core.pipeline.indexing import FAISSVectorStore
    from pathlib import Path

    # Properly instantiate FAISSVectorStore with required paths
    index_path = Path(vector_store_path) / "index.faiss"
    metadata_path = Path(vector_store_path) / "metadata.json"

    console.print(f"[bold]Loading vector store from:[/bold] {index_path}")
    store = FAISSVectorStore(
        index_path=index_path,
        metadata_path=metadata_path,
        similarity=settings.get("VECTOR_STORE_SIMILARITY_ALGORITHM", "cosine")
    )

    # Define embedding function
    def embed_fn(text):
        if isinstance(text, list):
            return client.get_embeddings(text)
        return client.get_embeddings([text])[0]

    # Create RAG request
    request = RAGRequest(
        question=question,
        primary_domain="default",  # Use default domain
        model=settings.get("MODEL", "gpt-4o-mini"),
        top_k=settings.get("TOP_K", 5)
    )

    # Track RAG process time
    start_time = time.time()
    with console.status("[bold green]Processing RAG request...") as status:
        # Process the question
        response = process_rag_request(request, embed_fn, str(vector_store_path))
    processing_time = time.time() - start_time

    # Display results in a nice formatted way
    from rich.panel import Panel
    from rich.markdown import Markdown

    # Show retrieved documents
    if response.documents:
        console.print("[bold]Retrieved Documents:[/bold]")
        for i, doc in enumerate(response.documents[:3]):
            console.print(Panel(
                f"[bold cyan]Document {i+1}[/bold cyan] (score: {doc.get('score', 'N/A')})\n\n"
                f"{doc.get('text', '')[:300]}...",
                title=f"Document {i+1}/{len(response.documents)}"
            ))

    # Show the answer
    console.print(Panel(
        Markdown(response.answer),
        title="AI Answer",
        subtitle=f"Processing time: {processing_time:.2f}s"
    ))

    # Show citations if available
    if response.citations:
        console.print(Panel(
            Markdown(response.citations),
            title="Citations"
        ))

    return response

def main():
    parser = argparse.ArgumentParser(description="Verbose Excel ETL and testing with rich visualization")
    parser.add_argument("--excel", "-e", required=True, help="Path to Excel file to process")
    parser.add_argument("--customer", "-c", default="coupa", help="Customer ID (default: coupa)")
    parser.add_argument("--extractor", "-x", default="cso_workflow", help="Extractor name to use")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Limit of rows to extract")
    parser.add_argument("--question", "-q", default="What information is in this dataset?", help="Test question to answer")
    parser.add_argument("--dataset-id", "-d", default="excel_verbose_test", help="Dataset ID for the vector store")
    parser.add_argument("--skip-qa", action="store_true", help="Skip the question answering phase")

    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold green]AI QnA Assistant - Verbose Excel ETL Test[/bold green]",
        subtitle="Shows detailed progress of each processing stage"
    ))

    # Run the ETL pipeline with enhanced visualization
    etl_results = execute_excel_etl(args.excel, args.customer, args.extractor, args.limit, args.dataset_id)

    if etl_results and not args.skip_qa:
        # Test question answering with detailed output
        test_question_answering(etl_results["vector_store_path"], args.question)

    console.print("[bold green]Test completed![/bold green]")
    return 0

if __name__ == "__main__":
    sys.exit(main())
