#!/usr/bin/env python3
"""
Gold-standard end-to-end pipeline test and diagnostics.
Covers loading, extraction, cleaning, chunking, embedding, exporting, indexing, retrieval, and RAG Q&A.
Provides rich diagnostics, timing, and summary output.
"""
import logging
import sys
import json
from pathlib import Path
import argparse
from typing import List
from core.config.settings import settings
from core.utils.component_registry import create_component_instance
import core.pipeline.extractors  # Ensure all extractors are registered
import core.pipeline.loaders
import customers.coupa.extractors.cso_workflow_extractor

# Optional: for RAG Q&A
from core.llm import refresh_client
from core.generation.rag_processor import process_rag_query, format_rag_response
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from tqdm import tqdm

# Setup rich console and logger
custom_theme = Theme({
    "stage1": "bold cyan",
    "stage2": "bold magenta",
    "stage3": "bold green",
    "stage4": "bold yellow",
    "stage5": "bold blue",
    "stage6": "bold red",
    "stage7": "bold white",
    "stage8": "bold bright_cyan",
    "stage9": "bold bright_magenta",
    "default": "white"
})
console = Console(theme=custom_theme)

# Replace logger handler with RichHandler for colored logs
logging.basicConfig(
    level=getattr(logging, "INFO".upper()),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[RichHandler(console=console, show_time=True, show_level=True, show_path=False)]
)
logger = logging.getLogger("pipeline_test")

stage_colors = [
    "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7", "stage8", "stage9"
]

# --- Diagnostics helpers ---
def print_stage(msg, stage_idx=0):
    color = stage_colors[stage_idx % len(stage_colors)]
    console.print(f"\n=== {msg} ===", style=color)

def print_sample(rows, n=2, stage_idx=0):
    color = stage_colors[stage_idx % len(stage_colors)]
    for i, row in enumerate(rows[:n]):
        console.print(f"Sample {i+1}: {str(row)[:200]}{'...' if len(str(row))>200 else ''}", style=color)


def main():
    parser = argparse.ArgumentParser(description="Gold-standard end-to-end pipeline test and diagnostics.")
    parser.add_argument("--customer", default="coupa", help="Customer name")
    parser.add_argument(
        "--dataset",
        default="/Users/ali.rafieefar/Documents/GitHub/ai_qna_assistant/customers/coupa/datasets/in-product-guides-Guide+Export/Workflow Steps.xlsb",
        help="Dataset file or directory (default: Coupa workflow XLSB)"
    )
    parser.add_argument(
        "--assets",
        default="/Users/ali.rafieefar/Documents/GitHub/ai_qna_assistant/customers/coupa/datasets/in-product-guides-Guide+Export/WorkflowSteps_unpacked",
        help="Assets folder (default: Coupa workflow assets)"
    )
    parser.add_argument("--loader", default="excel", help="Loader type (excel, html, parquet, etc.)")
    parser.add_argument("--extractor", default="cso_workflow", help="Extractor type")
    parser.add_argument("--cleaner", default="html", help="Cleaner type")
    parser.add_argument("--chunker", default="html", help="Chunker type")
    parser.add_argument("--embedder", default=None, help="Embedder type (default: from settings)")
    parser.add_argument("--exporter", default=None, help="Exporter type (optional)")
    parser.add_argument("--vectorstore", default=None, help="Vectorstore type (default: from settings)")
    parser.add_argument("--retriever", default=None, help="Retriever type (default: from settings)")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of docs (default: 5)")
    parser.add_argument("--export", action="store_true", help="Run exporter step")
    parser.add_argument("--run-qa", action="store_true", help="Run RAG Q&A on default questions")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--output", default="pipeline_output.json", help="Save output to file")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[RichHandler(console=console, show_time=True, show_level=True, show_path=False)])
    logger = logging.getLogger("pipeline_test")

    print_stage("STEP 1: Loading data", 0)
    from core.pipeline.loaders.excel_loader import ExcelLoader
    console.print(f"[yellow]ExcelLoader class: {ExcelLoader}[/yellow]")
    console.print(f"[yellow]ExcelLoader __init__ signature: {ExcelLoader.__init__}[/yellow]")
    if args.loader == "excel":
        try:
            from customers.coupa.loaders.cso_excel_loader import CSOExcelLoader
            loader = CSOExcelLoader(file_path=args.dataset, customer_id=args.customer)
            console.print("[green]Using CSOExcelLoader for Coupa customer.[/green]")
        except ImportError:
            from core.pipeline.loaders.excel_loader import ExcelLoader
            loader = ExcelLoader(file_path=args.dataset, customer_id=args.customer)
            console.print("[yellow]Falling back to generic ExcelLoader.[/yellow]")
            console.print(f"[red]ExcelLoader keyword args failed: {e}. Trying no-arg instantiation...[/red]")
            try:
                loader = ExcelLoader()
                console.print(f"[red]ExcelLoader instantiated with no arguments. You may need to set attributes manually.[/red]")
            except Exception as e2:
                console.print(f"[red]ExcelLoader failed with no arguments: {e2}")
                raise
        loaded_data = loader.load(args.dataset) if hasattr(loader, 'load') else None
        # If loaded_data is not a dict, but is a generator or list of pairs, convert to dict
        if loaded_data is not None and not isinstance(loaded_data, dict):
            loaded_data = list(loaded_data)
            console.print(f"[red]DEBUG: loaded_data type: {type(loaded_data)}, first element: {loaded_data[0] if loaded_data else 'EMPTY'}[/red]")
            if loaded_data and isinstance(loaded_data[0], tuple) and len(loaded_data[0]) == 2:
                loaded_data = dict(loaded_data)
            elif loaded_data and all(hasattr(df, 'columns') for df in loaded_data):
                loaded_data = {f"Sheet{i+1}": df for i, df in enumerate(loaded_data)}
            elif loaded_data and all(type(row).__name__ == "Row" for row in loaded_data):
                # Accept list of Row objects as valid
                pass
            elif not loaded_data:
                loaded_data = {}
            else:
                raise ValueError(f"Loaded data is not a dict, list of pairs, list of DataFrames, or list of Row objects. Type: {type(loaded_data)}")
        logger.info(f"Loaded {len(loaded_data) if loaded_data else 0} sheets from dataset {args.dataset}")
        if loaded_data:
            if isinstance(loaded_data, dict):
                for sheet_name, df in loaded_data.items():
                    logger.info(f"  Sheet '{sheet_name}': {len(df)} rows, columns: {list(df.columns)}")
                    if sheet_name.lower() == "guidedoc":
                        console.print(f"[bold yellow]DEBUG: guideDoc columns: {list(df.columns)}[/bold yellow]")
            elif isinstance(loaded_data, list):
                logger.info(f"Loaded_data is a list with {len(loaded_data)} elements (not a dict of sheets).")
                # If any Row has sheet_name 'guideDoc', print its metadata keys
                # Only print the first 3 guideDoc rows for debug
                guide_doc_debug_count = 0
                for row in loaded_data:
                    if hasattr(row, 'metadata') and row.metadata.get('sheet_name', '').lower() == 'guidedoc':
                        if guide_doc_debug_count < 3:
                            console.print(f"[bold yellow]DEBUG: Row from guideDoc: metadata keys: {list(row.metadata.keys())}, text: {row.text[:100]}...[/bold yellow]")
                            guide_doc_debug_count += 1
                        elif guide_doc_debug_count == 3:
                            console.print(f"[bold yellow]... (more guideDoc rows omitted) ...[/bold yellow]")
                            guide_doc_debug_count += 1
    else:
        from core.pipeline.loaders import create_loader
        loader = create_loader(args.loader, root_dir=args.dataset)
        loaded_data = list(loader) if hasattr(loader, '__iter__') else list(loader.load(args.dataset))
        logger.info(f"Loaded {len(loaded_data)} items from dataset {args.dataset} using loader '{args.loader}'")

    print_stage("STEP 2: Extraction", 1)
    from core.pipeline.extractors import get_extractor
    extractor_name = args.extractor
    ExtractorCls = get_extractor(extractor_name)
    console.print(f"[yellow]Extractor class: {ExtractorCls}[/yellow]")
    console.print(f"[yellow]Extractor __init__ signature: {ExtractorCls.__init__}[/yellow]")
    if args.loader == "excel":
        try:
            extractor = ExtractorCls(
                file_path=args.dataset,
                customer_id=args.customer,
                data=loaded_data
            )
        except TypeError as e:
            console.print(f"[red]Extractor keyword args failed: {e}. Trying no-arg instantiation...[/red]")
            try:
                extractor = ExtractorCls()
                console.print(f"[red]Extractor instantiated with no arguments. You may need to set attributes manually.[/red]")
            except Exception as e2:
                console.print(f"[red]Extractor failed with no arguments: {e2}")
                raise
        # --- Extraction diagnostics ---
        rows = []
        for i, row in enumerate(extractor.extract_rows() if hasattr(extractor, 'extract_rows') else []):
            # Print debug info for first 5 rows
            for i, row in enumerate(extractor.extract_rows() if hasattr(extractor, 'extract_rows') else []):
                if i < 5:
                    console.print(f"[bold magenta]EXTRACTOR DEBUG: Row {i+1}: type={type(row)}, attrs={dir(row)}[/bold magenta]")
                    if hasattr(row, 'metadata'):
                        console.print(f"[bold magenta]  metadata: {row.metadata}[/bold magenta]")
                        # Print guide title if present in metadata
                        guide_title = row.metadata.get('guide_title') or row.metadata.get('Name')
                        if guide_title is not None:
                            console.print(f"[bold magenta]  guide_title: {guide_title}[/bold magenta]")
                        # Warn if guide_title is missing or nan
                        if (guide_title is None or str(guide_title).lower() == 'nan'):
                            console.print(f"[red]WARNING: Row {i+1} has guide_title: nan or None! Row: {row}[/red]")
                    if hasattr(row, 'text'):
                        console.print(f"[bold magenta]  text: {str(row.text)[:120]}{'...' if len(str(row.text))>120 else ''}[/bold magenta]")
                rows.append(row)
                if len(rows) >= args.limit:
                    break
    else:
        extractor = ExtractorCls(file_path=args.dataset, customer_id=args.customer)
        rows = list(extractor.extract_rows())[:args.limit]
    logger.info(f"Extracted {len(rows)} rows using extractor '{extractor_name}'")
    print_sample(rows, stage_idx=1)

    print_stage("STEP 3: Cleaning", 2)
    cleaner = create_component_instance("cleaner", args.cleaner)
    cleaned_rows = [cleaner.clean(row) for row in rows]
    logger.info(f"Cleaned {len(cleaned_rows)} rows")
    print_sample(cleaned_rows, stage_idx=2)

    print_stage("STEP 4: Chunking", 3)
    chunker = create_component_instance("chunker", args.chunker)
    chunks = []
    for row in cleaned_rows:
        row_chunks = chunker.chunk(row)
        # --- Chunking diagnostics ---
        for i, chunk in enumerate(row_chunks):
            if i < 2:
                console.print(f"[bold green]CHUNK DEBUG: text: {str(chunk.text)[:120]}{'...' if len(str(chunk.text))>120 else ''} | metadata: {chunk.metadata}[/bold green]")
            if hasattr(chunk, 'text') and (not chunk.text or len(chunk.text.strip()) == 0):
                console.print(f"[red]WARNING: Empty chunk text! metadata: {chunk.metadata}[/red]")
            if hasattr(chunk, 'text') and len(chunk.text) > 4000:
                console.print(f"[red]WARNING: Huge chunk text ({len(chunk.text)} chars)! metadata: {chunk.metadata}[/red]")
        chunks.extend(row_chunks)
    logger.info(f"Chunked into {len(chunks)} chunks")
    print_sample(chunks, stage_idx=3)

    print_stage("STEP 5: Embedding", 4)
    embedder_type = args.embedder or settings.get("EMBEDDER_PROVIDER", "openai")
    embedder = create_component_instance("embedder", embedder_type)
    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    logger.info(f"Embedding parameters: EMBED_MODEL={settings.get('EMBED_MODEL')}, EMBEDDER_PROVIDER={settings.get('EMBEDDER_PROVIDER')}, VECTORSTORE_WORKERS={settings.get('VECTORSTORE_WORKERS')}, VECTOR_STORE_BATCH_SIZE={settings.get('VECTOR_STORE_BATCH_SIZE')}, BATCH_SIZE={settings.get('INDEX_DEFAULTS',{}) .get('BATCH_SIZE')}")
    logger.info(f"OPENAI_BASE_URL={getattr(settings, 'OPENAI_BASE_URL', None)} | TENANT_ID={getattr(settings, 'TENANT_ID', None)} | APPLICATION_NAME={getattr(settings, 'APPLICATION_NAME', None)} | USE_CASE={getattr(settings, 'USE_CASE', None)}")

    # Show preview of first 3 chunks and their metadata before embedding
    for i, (text, meta) in enumerate(zip(texts, metadatas)):
        if i < 3:
            console.print(f"[bold blue]EMBED PREVIEW: text: {str(text)[:120]}{'...' if len(str(text))>120 else ''} | metadata: {meta}[/bold blue]")

    # Show progress bar for embedding
    embeddings = []
    for emb in tqdm(embedder.embed_text(texts), desc="Embedding", unit="vec", ncols=80):
        embeddings.append(emb)
    logger.info(f"Embedded {len(embeddings)} chunks")

    # --- Show embedding parameters for debug ---
    logger.info(f"Embedding parameters: EMBED_MODEL={settings.get('EMBED_MODEL')}, EMBEDDER_PROVIDER={settings.get('EMBEDDER_PROVIDER')}, VECTORSTORE_WORKERS={settings.get('VECTORSTORE_WORKERS')}, VECTOR_STORE_BATCH_SIZE={settings.get('VECTOR_STORE_BATCH_SIZE')}, BATCH_SIZE={settings.get('INDEX_DEFAULTS',{}) .get('BATCH_SIZE')}")
    logger.info(f"OPENAI_BASE_URL={getattr(settings, 'OPENAI_BASE_URL', None)} | TENANT_ID={getattr(settings, 'TENANT_ID', None)} | APPLICATION_NAME={getattr(settings, 'APPLICATION_NAME', None)} | USE_CASE={getattr(settings, 'USE_CASE', None)}")

    if args.export and args.exporter:
        print_stage("STEP 6: Exporting", 5)
        exporter = create_component_instance("exporter", args.exporter)
        exporter.run(cleaned_rows)
        logger.info(f"Exported data using exporter '{args.exporter}'")

    print_stage("STEP 7: Indexing", 6)
    vectorstore_type = args.vectorstore or settings.get("VECTORSTORE_PROVIDER", "faiss")
    vector_store_id = f"{args.customer}_{Path(args.dataset).stem}"
    
    if vectorstore_type.lower() == "faiss":
        index_dir = Path("vector_store") / vector_store_id
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "index.faiss"
        metadata_path = index_dir / "metadata.jsonl"
        vector_store = create_component_instance(
            "vectorstore",
            vectorstore_type,
            index_path=index_path,
            metadata_path=metadata_path,
            similarity=settings.get("VECTORSTORE_SIMILARITY_ALGORITHM", "cosine")
        )
    else:
        vector_store = create_component_instance(
            "vectorstore",
            vectorstore_type,
            store_id=vector_store_id
        )
    vector_store.add(embeddings, metadatas, texts)
    logger.info(f"Indexed {len(embeddings)} embeddings in vector store '{vector_store_id}'")

    print_stage("STEP 8: Retrieval", 7)
    retriever_type = args.retriever or settings.get("VECTORSTORE_PROVIDER", "faiss")
    if retriever_type.lower() == "faiss":
        index_dir = Path("vector_store") / vector_store_id
        index_dir.mkdir(parents=True, exist_ok=True)
        retriever = create_component_instance(
            "retriever",
            retriever_type,
            index_dir=index_dir
        )
    else:
        retriever = create_component_instance(
            "retriever",
            retriever_type,
            store_id=vector_store_id
        )
    test_queries = [
        "What information is in this dataset?",
        "How does this process work?",
        "What steps are involved?"
    ]
    for query in test_queries:
        results = retriever.retrieve(query, top_k=3)
        logger.info(f"Retrieved {len(results)} docs for test query: {query}")
        for i, doc in enumerate(results, 1):
            logger.info(f"Result {i}: {doc}")

    if args.run_qa:
        print_stage("STEP 9: RAG Q&A (batch)", 8)
        refresh_client()
        default_questions = [
            "What is the main purpose of this dataset?",
            "List the key steps described.",
            "Summarize the workflow."
        ]
        for question in default_questions:
            rag_result = process_rag_query(
                question=question,
                top_k=3,
                index_dir=vector_store_id
            )
            answer = format_rag_response(rag_result)
            sources = [doc.get('source', 'Unknown') for doc in rag_result.get('documents', [])]
            print(f"Q: {question}\nA: {answer}\nSources: {sources}")
            print(f"Timing: {rag_result.get('timing', {})}")
            print(f"Citations: {rag_result.get('citations', '')}\n")

    # Save summary
    if args.output:
        import datetime
        output_data = {
            "date": datetime.datetime.now().isoformat(),
            "dataset": args.dataset,
            "customer": args.customer,
            "steps": [
                "loading", "extraction", "cleaning", "chunking", "embedding", "exporting", "indexing", "retrieval", "rag_qa"
            ],
            "metrics": {
                "rows_extracted": len(rows),
                "chunks_created": len(chunks),
                "embeddings_created": len(embeddings)
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Summary saved to {args.output}")

    console.print("\nPipeline test completed successfully.", style="bold green")

if __name__ == "__main__":
    main()
