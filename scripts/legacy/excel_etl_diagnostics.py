#!/usr/bin/env python
"""
Step-by-step diagnostic tool for Excel ETL processing.

This script provides individual functions to test each part of the Excel ETL pipeline
separately, making it easier to diagnose issues and see detailed output at each stage.

Example usage:
- To list available extractors:
    python scripts/excel_etl_diagnostics.py --list-extractors

- To extract from Excel:
    python scripts/excel_etl_diagnostics.py --step extract --excel PATH_TO_EXCEL --extractor cso_workflow --limit 5

- To build index from previously extracted documents:
    python scripts/excel_etl_diagnostics.py --step index --customer coupa --dataset-id test_excel

- To test question answering:
    python scripts/excel_etl_diagnostics.py --step qa --vector-store PATH_TO_VECTOR_STORE --question "What is in this dataset?"
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to import from the root package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup to ensure proper module resolution
from core.utils.component_registry import create_component_instance, available
from core.config.settings import settings
from core.llm import get_llm_client
from core.pipeline.indexing.indexer import build_index_from_documents
from core.generation.rag_processor import process_rag_request, RAGRequest
from core.utils.component_registry import get as get_component
from core.services.customer_service import customer_service
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Ensure customer package is imported to register components
import customers.coupa  # noqa: F401

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def print_header(message: str):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def print_doc_sample(doc, index: int):
    """Print a formatted sample of a document"""
    print(f"\n--- Document {index} ---")
    if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
        # LangChain Document
        print(f"Content (first 200 chars): {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
    elif isinstance(doc, dict):
        # Dictionary document
        if "text" in doc:
            print(f"Text (first 200 chars): {doc.get('text', '')[:200]}...")
        if "metadata" in doc:
            print(f"Metadata: {doc.get('metadata', {})}")
    else:
        print(f"Unknown document format: {type(doc)}")
        print(f"Content: {doc}")

def save_documents(documents: List[Any], output_file: str):
    """Save extracted documents to a JSON file for inspection or later use"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert documents to serializable format if needed
    serializable_docs = []
    for doc in documents:
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            # LangChain Document
            serializable_docs.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
        elif isinstance(doc, dict):
            # Dictionary document (already serializable)
            serializable_docs.append(doc)

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_docs, f, indent=2, ensure_ascii=False, default=str)

    print(f"Saved {len(serializable_docs)} documents to {output_path}")
    return output_path

def load_documents(input_file: str) -> List[Document]:
    """Load documents from a JSON file, converting to LangChain Documents"""
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return []

    # Load from file
    with open(input_path, 'r', encoding='utf-8') as f:
        serializable_docs = json.load(f)

    # Convert to LangChain Documents
    documents = []
    for doc_dict in serializable_docs:
        if "text" in doc_dict and "metadata" in doc_dict:
            doc = Document(
                page_content=doc_dict["text"],
                metadata=doc_dict["metadata"]
            )
            documents.append(doc)

    print(f"Loaded {len(documents)} documents from {input_path}")
    return documents

# ---------------------------------------------------------
# Step 1: List available extractors
# ---------------------------------------------------------

def list_available_extractors():
    """List all available extractors from the component registry"""
    print_header("Available Extractors")
    extractors = available("extractor")

    if not extractors:
        print("No extractors found in registry")
        return

    print(f"Found {len(extractors)} registered extractors:")
    for i, extractor_name in enumerate(extractors, 1):
        print(f"  {i}. {extractor_name}")

    print("\nTo use an extractor, run:")
    print("  python scripts/excel_etl_diagnostics.py --step extract --excel PATH_TO_EXCEL --extractor EXTRACTOR_NAME")

# ---------------------------------------------------------
# Step 2: Extract data from Excel
# ---------------------------------------------------------

def extract_from_excel(
    excel_path: str,
    extractor_name: str,
    customer_id: str = "coupa",
    limit: int = 5,
    output_file: Optional[str] = None
) -> List[Document]:
    """Extract data from Excel file using the specified extractor"""
    print_header(f"Extracting Data from Excel using '{extractor_name}'")
    print(f"Excel file: {excel_path}")
    print(f"Customer ID: {customer_id}")
    print(f"Limit: {limit}")

    excel_path_obj = Path(excel_path)
    if not excel_path_obj.exists():
        print(f"Error: Excel file not found: {excel_path}")
        return []

    try:
        # Get the extractor component from registry
        extractor_class = get_component("extractor", extractor_name)
        print(f"Using extractor class: {extractor_class.__name__}")

        # Initialize extractor
        extractor = create_component_instance(
            "extractor",
            extractor_name,
            file_path=excel_path_obj,
            customer_id=customer_id
        )
        print(f"Extractor instance created: {type(extractor).__name__}")

        # Extract rows one by one
        print("\nExtracting rows...")
        rows = []
        start_time = time.time()

        for i, row_data in enumerate(extractor.extract_rows()):
            print(f"Processing row {i+1}...", end="\r")
            rows.append(row_data)
            if i + 1 >= limit:
                print(f"Reached extraction limit of {limit} rows.")
                break

        extraction_time = time.time() - start_time
        print(f"\nExtracted {len(rows)} rows in {extraction_time:.2f} seconds")

        # Show sample of raw data
        if rows:
            print("\n=== Sample of raw extracted data ===")
            sample_row = rows[0]
            print(f"Raw data keys: {list(sample_row.keys())}")
            if 'text' in sample_row:
                print(f"Text preview: {str(sample_row['text'])[:200]}...")
            if 'metadata' in sample_row:
                print(f"Metadata: {sample_row['metadata']}")
        else:
            print("No rows extracted!")
            return []

        # Convert rows to Langchain Documents
        print("\nConverting to Langchain Documents...")
        documents = []

        for i, row_dict in enumerate(rows):
            try:
                if 'text' not in row_dict or 'metadata' not in row_dict:
                    print(f"Row {i+1} missing 'text' or 'metadata' key - skipping")
                    continue

                doc = Document(
                    page_content=str(row_dict["text"]),
                    metadata=row_dict["metadata"]
                )
                documents.append(doc)
            except Exception as e:
                print(f"Error converting row {i+1}: {str(e)}")

        print(f"Successfully converted {len(documents)} rows to Langchain Documents")

        # Show document samples
        if documents:
            print("\n=== Document Samples ===")
            for i, doc in enumerate(documents[:3]):
                print_doc_sample(doc, i+1)

        # Save to file if requested
        if output_file:
            output_path = save_documents(documents, output_file)
            print(f"\nDocuments saved to {output_path} for future use")

        return documents

    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        traceback.print_exc()
        return []

# ---------------------------------------------------------
# Step 3: Build index from documents
# ---------------------------------------------------------

def build_index(
    documents: List[Document],
    customer_id: str = "coupa",
    dataset_id: str = "test_excel",
    from_file: Optional[str] = None
) -> Optional[str]:
    """Build vector index from documents"""
    print_header(f"Building Vector Index for '{dataset_id}'")

    # Load documents from file if specified
    if from_file:
        documents = load_documents(from_file)
        if not documents:
            return None

    if not documents:
        print("Error: No documents to index")
        return None

    print(f"Building index for {len(documents)} documents...")
    print(f"Customer: {customer_id}")
    print(f"Dataset ID: {dataset_id}")

    try:
        # Track time
        start_time = time.time()

        # Build the index
        vector_store_path = build_index_from_documents(
            docs=documents,
            customer_id=customer_id,
            dataset_id=dataset_id
        )

        build_time = time.time() - start_time
        print(f"\nIndex built successfully in {build_time:.2f} seconds")
        print(f"Vector store path: {vector_store_path}")

        # List files in the vector store
        if vector_store_path:
            vector_dir = Path(vector_store_path)
            if vector_dir.exists():
                files = list(vector_dir.glob("*"))
                print("\nVector store files:")
                for file in files:
                    size_kb = file.stat().st_size / 1024
                    print(f"  - {file.name} ({size_kb:.2f} KB)")

        return vector_store_path

    except Exception as e:
        print(f"Error building index: {str(e)}")
        traceback.print_exc()
        return None

# ---------------------------------------------------------
# Step 4: Test question answering
# ---------------------------------------------------------

def test_question_answering(
    vector_store_path: str,
    question: str = "What information is in this dataset?"
):
    """Test question answering against a vector store"""
    print_header("Testing Question Answering")
    print(f"Vector store: {vector_store_path}")
    print(f"Question: \"{question}\"")

    try:
        # Check if vector store exists
        vs_path = Path(vector_store_path)

        # Check for different possible file extensions (json or jsonl)
        index_path = vs_path / "index.faiss"
        metadata_path = vs_path / "metadata.json"
        metadata_path_alt = vs_path / "metadata.jsonl"

        # Print more detailed debugging information
        print(f"\nDebug: Checking for vector store files...")
        print(f"Vector store directory exists: {vs_path.exists()}")
        if vs_path.exists():
            print(f"Files in directory:")
            for f in vs_path.glob("*"):
                print(f"  - {f.name} ({f.stat().st_size / 1024:.2f} KB)")

        print(f"Index file exists: {index_path.exists()}")
        print(f"Metadata file (json) exists: {metadata_path.exists()}")
        print(f"Metadata file (jsonl) exists: {metadata_path_alt.exists()}")

        # Use the first metadata file that exists
        if metadata_path.exists():
            actual_metadata_path = metadata_path
        elif metadata_path_alt.exists():
            actual_metadata_path = metadata_path_alt
            print(f"Using alternative metadata file format (.jsonl)")
        else:
            print(f"Error: No metadata file found in {vs_path}")
            return None

        if not index_path.exists():
            print(f"Error: Index file not found at {index_path}")
            return None

        # Initialize OpenAI client
        client = get_llm_client()
        print(f"LLM provider: {settings.get('LLM_PROVIDER', 'openai')}")

        # Initialize vector store for local testing
        from core.pipeline.indexing import FAISSVectorStore

        # Properly instantiate FAISSVectorStore with required paths
        print("\nInitializing vector store...")
        store = FAISSVectorStore(
            index_path=index_path,
            metadata_path=actual_metadata_path,
            similarity=settings.get("VECTOR_STORE_SIMILARITY_ALGORITHM", "cosine")
        )
        print("Vector store initialized successfully.")

        # Define embedding function
        def embed_fn(text):
            if isinstance(text, list):
                return client.get_embeddings(text)
            return client.get_embeddings([text])[0]

        # Create RAG request
        request = RAGRequest(
            question=question,
            primary_domain="default",
            model=settings.get("MODEL", "gpt-4o-mini"),
            top_k=settings.get("TOP_K", 5)
        )

        # Process the question
        print("\nProcessing RAG request...")
        start_time = time.time()

        # Important: Pass just the directory path instead of the full index file path
        # This matches the expectation in the retrieval system
        response = process_rag_request(request, embed_fn, str(vs_path))

        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")

        # Show retrieved documents
        if response.documents:
            print(f"\nRetrieved {len(response.documents)} documents:")
            for i, doc in enumerate(response.documents[:3], 1):
                print(f"\n--- Retrieved Document {i}/{len(response.documents)} ---")
                print(f"Score: {doc.get('score', 'N/A')}")
                print(f"Content: {doc.get('text', '')[:300]}...")
        else:
            print("No documents retrieved")

        # Show the answer
        print("\n=== AI Answer ===")
        print(response.answer)

        # Show citations if available
        if response.citations:
            print("\n=== Citations ===")
            print(response.citations)

        return response

    except Exception as e:
        print(f"Error in question answering: {str(e)}")
        traceback.print_exc()
        return None

# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step-by-step diagnostic tool for Excel ETL")
    parser.add_argument("--step", choices=["extract", "index", "qa", "all"],
                        help="Processing step to execute")
    parser.add_argument("--list-extractors", action="store_true",
                        help="List available extractors and exit")
    parser.add_argument("--excel", help="Path to Excel file")
    parser.add_argument("--customer", default="coupa",
                        help="Customer ID (default: coupa)")
    parser.add_argument("--extractor", help="Extractor name to use")
    parser.add_argument("--limit", type=int, default=5,
                        help="Limit of rows to extract (default: 5)")
    parser.add_argument("--dataset-id", default="test_excel",
                        help="Dataset ID for the vector store (default: test_excel)")
    parser.add_argument("--vector-store",
                        help="Path to vector store for QA")
    parser.add_argument("--question", default="What information is in this dataset?",
                        help="Question to answer (default: 'What information is in this dataset?')")
    parser.add_argument("--output-file",
                        help="Save extracted documents to this file (JSON format)")
    parser.add_argument("--input-file",
                        help="Load documents from this file instead of extracting")

    args = parser.parse_args()

    # Validate arguments
    if args.list_extractors:
        return list_available_extractors()

    if not args.step:
        parser.print_help()
        return 1

    # Execute the selected step
    if args.step == "extract" or args.step == "all":
        if not args.excel:
            print("Error: --excel argument is required for extraction")
            return 1
        if not args.extractor:
            print("Error: --extractor argument is required for extraction")
            return 1

        documents = extract_from_excel(
            args.excel,
            args.extractor,
            args.customer,
            args.limit,
            args.output_file
        )

        if args.step == "all" and documents:
            vector_store_path = build_index(
                documents,
                args.customer,
                args.dataset_id
            )

            if vector_store_path:
                test_question_answering(
                    vector_store_path,
                    args.question
                )

    elif args.step == "index":
        documents = []
        if args.input_file:
            build_index(
                [],
                args.customer,
                args.dataset_id,
                from_file=args.input_file
            )
        else:
            print("Error: --input-file argument is required for indexing without extraction")
            return 1

    elif args.step == "qa":
        if not args.vector_store:
            print("Error: --vector-store argument is required for question answering")
            return 1

        test_question_answering(
            args.vector_store,
            args.question
        )

    return 0

if __name__ == "__main__":
    sys.exit(main())
