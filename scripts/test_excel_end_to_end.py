#!/usr/bin/env python
"""
End-to-end test for the ai_qna_assistant with Excel dataset.
This script follows the golden checklist guidelines to test an Excel dataset through
the entire pipeline (extraction, indexing, RAG) while respecting the project architecture.

python scripts/test_excel_end_to_end.py --excel "/Users/ali.rafieefar/Documents/GitHub/ai_backup/0205_working_exporter_FAISS_vector/data/datasets/in-product-guides-Guide+Export/Workflow Steps.xlsb" --customer coupa --extractor cso_workflow --limit 5

"""

import argparse
import logging
import sys
from pathlib import Path

import pytest  # Added for pytest.skip

# Add parent directory to path to import from the root package
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils.component_registry import create_component_instance
from core.config.settings import settings
from core.config.base_paths import (
    find_project_root
)
from core.llm import get_llm_client
from core.pipeline.indexing.indexer import build_index_from_documents
from core.generation.rag_processor import process_rag_request, RAGRequest
from core.utils.component_registry import get as get_component
from core.services.customer_service import customer_service
from langchain.schema import Document

# Ensure customer package is imported to register components
import customers.coupa  # noqa: F401

# Ensure all LLM providers are registered before use (fixes KeyError: 'llm')

# --- NOTEBOOK/INTERACTIVE SUPPORT ---
# Only run this block if in a notebook or interactive environment
import sys


def _in_notebook():
    try:
        from IPython.core.getipython import get_ipython
        if 'IPKernelApp' in get_ipython().config:  # type: ignore
            return True
    except Exception:
        pass
    return False

if _in_notebook():
    # Use project root for notebook paths
    project_root = find_project_root()
    sys.argv = [
        "test_excel_end_to_end.py",
        "--excel", str(project_root / "customers/coupa/datasets/in-product-guides-Guide+Export/Workflow Steps.xlsb"),
        "--customer", "coupa",
        "--extractor", "cso_workflow",
        "--limit", "5"
    ]
    print("Simulated sys.argv for notebook execution:", sys.argv)
    print("LLM providers registered.")
# --- END NOTEBOOK/INTERACTIVE SUPPORT ---

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from logging.INFO to logging.DEBUG
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s", # Added filename and lineno
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def execute_excel_test_logic(excel_file_str: str, customer: str, extractor: str, limit: int, question_text: str, dataset_id_str: str) -> bool:
    """Core logic for the Excel end-to-end test."""
    excel_path = Path(excel_file_str)
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        return False

    logger.info(f"Starting end-to-end test for {customer} with {extractor}")
    logger.info(f"Using Excel file: {excel_path}")

    # Configure settings
    if not configure_test_settings(customer, excel_path, limit):
        logger.error("Failed to configure test settings")
        return False

    # Extract data
    documents = extract_data(customer, excel_path, extractor, limit)
    logger.info(f"extract_data returned {len(documents) if documents else 0} documents.")
    if documents:
        logger.info(f"Sample document[0]: {documents[0] if len(documents) > 0 else 'N/A'}")
    else:
        logger.error("Failed to extract data from Excel or no documents returned.")
        return False

    # --- DEBUGGING: Print extracted rows and documents for inspection ---

    # Run extraction step manually for inspection
    customer_id = "coupa"
    excel_path = Path(excel_file_str)  # Use the same path that was passed in
    extractor_name = "cso_workflow"
    limit = 5

    # Create extractor using the registry helper (gold standard)
    try:
        extractor = create_component_instance(
            "extractor",
            extractor_name,
            file_path=excel_path,
            customer_id=customer_id
        )
    except Exception as e:
        logger.error(f"Failed to create extractor instance: {e}")
        return False

    rows = []
    for i, row_data in enumerate(extractor.extract_rows()):
        print(f"Raw row {i}: {row_data}")
        rows.append(row_data)
        if i + 1 >= limit:
            break

    print(f"\nExtracted {len(rows)} rows.")
    if rows:
        print("Sample extracted row:", rows[0])

    # Convert to Langchain Documents
    from langchain.schema import Document
    documents = []
    for i, row_dict in enumerate(rows):
        if 'text' in row_dict and 'metadata' in row_dict:
            doc = Document(page_content=str(row_dict["text"]), metadata=row_dict["metadata"])
            documents.append(doc)
            print(f"Document {i}: {doc}")
        else:
            print(f"Row {i} missing 'text' or 'metadata': {row_dict}")

    print(f"\nConverted {len(documents)} rows to Langchain Documents.")
    if documents:
        print("Sample Document:", documents[0])
    else:
        print("No valid Documents created. Check extraction output above.")


    # Build index
    vector_store_path = build_index_for_test(customer, documents, dataset_id_str)
    logger.info(f"build_index_for_test returned: {vector_store_path}")
    if not vector_store_path:
        logger.error("Failed to build index")
        return False

    # Test question answering
    qa_result = _execute_question_answering_test(vector_store_path, question_text)
    logger.info(f"_execute_question_answering_test returned: {qa_result}")
    if not qa_result:
        logger.error("Failed to test question answering")
        return False

    logger.info("End-to-end test logic completed successfully")
    return True


def configure_test_settings(customer_id, excel_path, limit=5):
    """Configure settings for the test run"""
    logger.info(f"Configuring test settings for customer: {customer_id}")

    # Load customer configuration
    customer_service.load_customer(customer_id)

    # Define overrides with keys matching AppConfig field names (assumed lowercase)
    # Based on DEFAULT_CONFIG["UPPERCASE_KEY"] -> field_name: lowercase_key
    field_overrides = {
        "export_limit": limit,  # Assuming AppConfig.export_limit from DEFAULT_CONFIG["EXPORT_LIMIT"]
        "test_excel_path": str(excel_path), # Assuming AppConfig.test_excel_path from DEFAULT_CONFIG["TEST_EXCEL_PATH"]
        "enable_citations": True # AppConfig.enable_citations is confirmed by the settings.py excerpt
    }

    logger.info("Applying test-specific setting overrides:")
    for key, value in field_overrides.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
            logger.info(f"  settings.{key} = {getattr(settings, key)}")
        else:
            logger.error(f"  Error: Setting attribute '{key}' does not exist on AppConfig. Override failed for this key.")
            # You might want to raise an error here if the override is critical for the test
            # For now, it logs an error and continues.

    # Log the final state of these specific settings for verification
    # This helps confirm if the settings were applied as expected.
    final_settings_values = {k: getattr(settings, k, f"N/A - {k} not found") for k in field_overrides.keys()}
    logger.info(f"Final settings values after override attempt: {final_settings_values}")
    return True


def extract_data(customer_id, excel_path, extractor_name, limit=5):
    """Extract data from Excel using the appropriate extractor"""
    logger.info(f"Extracting data from Excel: {excel_path}")

    try:
        # Get the appropriate extractor from the component registry
        logger.debug(f"Getting component 'extractor' with name '{extractor_name}'")
        extractor_class = get_component("extractor", extractor_name)
        logger.info(f"Using extractor class: {extractor_class.__name__}")

        # Always instantiate with config dict for maximum compatibility
        try:
            logger.debug(f"Instantiating extractor with parameters: file_path={excel_path}, customer_id={customer_id}")
            extractor = extractor_class(**{"file_path": excel_path, "customer_id": customer_id})
            logger.debug("Extractor successfully instantiated")
        except Exception as e:
            logger.error(f"Failed to instantiate extractor {extractor_class.__name__} with config dict: {e}")
            return []
        logger.info("Extractor instance created.")

        # Extract rows with limit
        row_count = 0
        rows = []
        logger.info("Starting to iterate through extractor.extract_rows()...")
        for i, row_data in enumerate(extractor.extract_rows()):
            logger.debug(f"Raw row {i} data type: {type(row_data)}")
            if not isinstance(row_data, dict):
                logger.error(f"Row {i} from extractor is not a dict: {type(row_data)}. Skipping.")
                continue

            # Log the keys in the row data
            logger.debug(f"Row {i} keys: {row_data.keys()}")
            rows.append(row_data)
            row_count += 1
            if row_count >= limit:
                logger.info(f"Reached extraction limit of {limit} rows.")
                break
        logger.info(f"Finished iterating through extractor.extract_rows(). Extracted {row_count} raw rows.")

        # Display sample output for verification
        if rows:
            logger.info("Sample extracted raw row (first one if available):")
            sample = rows[0]
            logger.info(f"  Raw sample keys: {list(sample.keys())}")
            logger.info(f"  Raw sample text preview (if 'text' key exists): {str(sample.get('text','N/A'))[:100]}...")
            logger.info(f"  Raw sample metadata preview (if 'metadata' key exists): {str(sample.get('metadata','N/A'))[:100]}...")
            logger.info(f"  Raw sample assets preview (if 'assets' key exists): {str(sample.get('assets','N/A'))[:100]}...")
        else:
            logger.warning("No rows were extracted by the extractor or processing failed before Document conversion.")

        # Convert rows to Documents for indexing
        documents = []
        logger.info(f"Attempting to convert {len(rows)} extracted rows to Langchain Documents.")
        for i, row_dict in enumerate(rows):
            try:
                if 'text' not in row_dict or 'metadata' not in row_dict:
                    logger.error(f"Row {i} is missing 'text' or 'metadata' key. Row data: {str(row_dict)[:200]}. Skipping Document conversion for this row.")
                    continue
                doc = Document(
                    page_content=str(row_dict["text"]), # Ensure page_content is string
                    metadata=row_dict["metadata"]
                )
                documents.append(doc)
                logger.debug(f"Successfully converted row {i} to Document.")
            except KeyError as ke:
                logger.error(f"KeyError converting row {i} to Document: {ke}. Row data: {str(row_dict)[:200]}")
            except Exception as e_doc:
                logger.error(f"Unexpected error converting row {i} to Document: {e_doc}. Row data: {str(row_dict)[:200]}")

        logger.info(f"Successfully converted {len(documents)} rows to Langchain Documents.")
        if not documents and rows: # If rows were extracted but no documents were made
            logger.error("Rows were extracted, but no Langchain Documents could be created. Check for missing 'text' or 'metadata' keys in extracted rows.")
        elif not documents:
            logger.error("No rows were extracted and no documents created.")
        else:
            logger.info(f"First document preview - content: {documents[0].page_content[:100]}")
            logger.info(f"First document metadata keys: {documents[0].metadata.keys()}")

        return documents
    except Exception as e:
        logger.error(f"Error extracting data: {e}", exc_info=True)
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return []


def build_index_for_test(customer_id, documents, dataset_id="excel_test"):
    """Build the FAISS index for vector search"""
    logger.info("Building FAISS index for vector search...")
    try:
        # Build index
        vector_store_path = build_index_from_documents(
            docs=documents,
            customer_id=customer_id,
            dataset_id=dataset_id
        )

        logger.info(f"Index built successfully at {vector_store_path}")
        return vector_store_path
    except Exception as e:
        logger.error(f"Error building index: {e}", exc_info=True)
        return None


def _execute_question_answering_test(vector_store_path, question):
    """Test the question answering system"""
    logger.info(f"Testing question: '{question}'")

    try:
        # Initialize OpenAI client
        client = get_llm_client()

        # Directly instantiate FAISSVectorStore for retrieval
        from core.pipeline.indexing import FAISSVectorStore
        from pathlib import Path
        try:
            # Properly instantiate FAISSVectorStore with required paths
            index_path = Path(vector_store_path) / "index.faiss"
            metadata_path = Path(vector_store_path) / "metadata.json"
            store = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                similarity=settings.get("VECTOR_STORE_SIMILARITY_ALGORITHM", "cosine")
            )
        except Exception as e:
            logger.error(f"Could not instantiate FAISSVectorStore. Error: {e}")
            return False
        def retrieve(query, top_k=5):
            return store.search(query, k=top_k)

        # Create RAG request
        request = RAGRequest(
            question=question,
            primary_domain="default",  # Use default domain
            model=settings.get("MODEL", "gpt-4o-mini"),
            top_k=settings.get("TOP_K", 5)
        )

        # Get embedding function from the client
        def embed_fn(text):
            if isinstance(text, list):
                return client.get_embeddings(text)
            return client.get_embeddings([text])[0]

        # Process the question
        response = process_rag_request(request, embed_fn, str(vector_store_path))

        # Display results
        logger.info("=== Question Answer Results ===")
        logger.info(f"Question: {question}")
        logger.info(f"Answer: {response.answer}")
        logger.info("\nUsed documents:")
        for i, doc in enumerate(response.documents[:3]):
            logger.info(f"Doc {i+1}: {doc['text'][:100]}...")

        return True
    except Exception as e:
        logger.error(f"Error in question answering: {e}", exc_info=True)
        return False


def list_available_extractors():
    """List all available extractors from the component registry"""
    from core.utils.component_registry import available
    return available("extractor")


def test_excel_pipeline_default_run():
    """Pytest discoverable test function for the Excel E2E pipeline."""
    # Get the script directory and construct path to the Excel file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    excel_path = project_root / "customers" / "coupa" / "datasets" / "in-product-guides-Guide+Export" / "Workflow Steps.xlsb"

    if not excel_path.exists():
        pytest.skip(f"Test Excel file not found at '{excel_path}'. "
                    "Please create this file to run the test.")

    logger.info(f"Running pytest: test_excel_pipeline_default_run with {excel_path}")

    success = execute_excel_test_logic(
        excel_file_str=str(excel_path),
        customer="coupa",  # Or a dedicated test customer
        extractor="cso_workflow",  # Or a relevant test extractor
        limit=3,  # Keep limit small for faster tests
        question_text="What information is in this dataset?",
        dataset_id_str="pytest_excel_e2e"
    )
    assert success, "Excel end-to-end test logic failed"


def main():
    parser = argparse.ArgumentParser(description="End-to-end test for Excel data processing")
    parser.add_argument("--excel", "-e", required=True, help="Path to Excel file to process")
    parser.add_argument("--customer", "-c", default="coupa", help="Customer ID (default: coupa)")
    parser.add_argument("--extractor", "-x", default="cso_workflow", help="Extractor name to use")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Limit of rows to extract")
    parser.add_argument("--question", "-q", default="What information is in this dataset?", help="Test question to answer")
    parser.add_argument("--list-extractors", action="store_true", help="List available extractors and exit")
    parser.add_argument("--dataset-id", "-d", default="excel_test", help="Dataset ID for the vector store")

    args = parser.parse_args()

    if args.list_extractors:
        logger.info("Available extractors:")
        for extractor_name in list_available_extractors():
            logger.info(f"- {extractor_name}")
        return 0

    if not execute_excel_test_logic(args.excel, args.customer, args.extractor, args.limit, args.question, args.dataset_id):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
