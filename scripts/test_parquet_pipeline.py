#!/usr/bin/env python3
"""
End-to-end pipeline test for parquet files containing link and content columns.
Covers loading, extraction, cleaning, chunking, embedding, indexing, retrieval, and RAG Q&A.
Provides rich diagnostics, timing, and summary output.
"""
import logging
import sys
import json
import time
import os
from pathlib import Path
import argparse
from typing import List, Dict, Any
import pandas as pd
from core.config.settings import settings
from core.utils.component_registry import create_component_instance
import core.pipeline.extractors  # Ensure all extractors are registered
import core.pipeline.loaders

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

# Find the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Replace logger handler with RichHandler for colored logs
logging.basicConfig(
    level=getattr(logging, "INFO".upper()),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[RichHandler(console=console, show_time=True, show_level=True, show_path=False)]
)
logger = logging.getLogger("parquet_pipeline_test")

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

class ParquetLoader:
    """
    Loader for parquet files that contain 'link' and 'content' columns.
    """
    def __init__(self, file_path: str, **kwargs):
        self.file_path = file_path
        self.logger = logging.getLogger("ParquetLoader")

    def load(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Load data from a parquet file
        """
        self.logger.info(f"Loading parquet file: {self.file_path}")
        df = pd.read_parquet(self.file_path)

        # Validate columns
        required_columns = ['link', 'content']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in parquet file: {missing_columns}")

        # Convert to list of dictionaries
        records = df.to_dict('records')
        if limit:
            records = records[:limit]

        self.logger.info(f"Loaded {len(records)} records from {self.file_path}")
        return records

class HTMLExtractor:
    """
    Simple extractor for HTML content that's already in the content column
    """
    def __init__(self, **kwargs):
        self.logger = logging.getLogger("HTMLExtractor")

    def extract(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract content from HTML data that's already available
        """
        self.logger.info(f"Processing {len(data)} HTML documents")

        extracted_data = []
        for item in data:
            extracted_item = {
                'url': item.get('link', ''),
                'title': self._extract_title(item.get('content', '')),
                'content': item.get('content', ''),
                'metadata': {
                    'source_url': item.get('link', ''),
                    'extraction_method': 'html_extractor'
                }
            }
            extracted_data.append(extracted_item)

        self.logger.info(f"Extracted {len(extracted_data)} documents")
        return extracted_data

    def _extract_title(self, html_content: str) -> str:
        """
        Extract title from HTML content
        """
        import re
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1)
        return "No Title Found"

def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline test for parquet files.")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "customers/coupa/datasets/product_documentation_coupa_compass_export.parquet"),
        help="Parquet file path containing 'link' and 'content' columns"
    )
    parser.add_argument("--customer", default="coupa", help="Customer name")
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
    parser.add_argument("--output", default="parquet_pipeline_output.json", help="Save output to file")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[RichHandler(console=console, show_time=True, show_level=True, show_path=False)])
    logger = logging.getLogger("parquet_pipeline_test")

    # Output capture for diagnostics
    pipeline_output = {
        "stages": {},
        "timings": {},
        "counts": {},
        "config": vars(args),
    }

    start_time = time.time()

    # --- STEP 1: Loading data ---
    print_stage("STEP 1: Loading data", 0)
    t0 = time.time()

    loader = ParquetLoader(args.dataset)
    raw_docs = loader.load(limit=args.limit)

    t1 = time.time()
    pipeline_output["timings"]["loading"] = t1 - t0
    pipeline_output["counts"]["raw_docs"] = len(raw_docs)

    print_sample(raw_docs, n=1, stage_idx=0)
    console.print(f"Loaded {len(raw_docs)} documents in {t1-t0:.2f} seconds", style="stage1")

    # --- STEP 2: Extraction ---
    print_stage("STEP 2: Extraction", 1)
    t0 = time.time()

    extractor = HTMLExtractor()
    extracted_docs = extractor.extract(raw_docs)

    t1 = time.time()
    pipeline_output["timings"]["extraction"] = t1 - t0
    pipeline_output["counts"]["extracted_docs"] = len(extracted_docs)

    print_sample(extracted_docs, n=1, stage_idx=1)
    console.print(f"Extracted {len(extracted_docs)} documents in {t1-t0:.2f} seconds", style="stage2")

    # --- STEP 3: Cleaning ---
    print_stage("STEP 3: Cleaning", 2)
    t0 = time.time()

    cleaner_type = args.cleaner
    cleaner = create_component_instance("cleaner", cleaner_type)
    cleaned_docs = cleaner.clean(extracted_docs)

    t1 = time.time()
    pipeline_output["timings"]["cleaning"] = t1 - t0
    pipeline_output["counts"]["cleaned_docs"] = len(cleaned_docs)

    print_sample(cleaned_docs, n=1, stage_idx=2)
    console.print(f"Cleaned {len(cleaned_docs)} documents in {t1-t0:.2f} seconds", style="stage3")

    # --- STEP 4: Chunking ---
    print_stage("STEP 4: Chunking", 3)
    t0 = time.time()

    chunker_type = args.chunker
    chunker = create_component_instance("chunker", chunker_type)
    chunks = chunker.chunk(cleaned_docs)

    t1 = time.time()
    pipeline_output["timings"]["chunking"] = t1 - t0
    pipeline_output["counts"]["chunks"] = len(chunks)

    print_sample(chunks, n=1, stage_idx=3)
    console.print(f"Created {len(chunks)} chunks in {t1-t0:.2f} seconds", style="stage4")

    # --- STEP 5: Embedding ---
    print_stage("STEP 5: Embedding", 4)
    t0 = time.time()

    embedder_type = args.embedder or settings.embedder
    embedder = create_component_instance("embedder", embedder_type)
    embedded_chunks = embedder.embed(chunks)

    t1 = time.time()
    pipeline_output["timings"]["embedding"] = t1 - t0
    pipeline_output["counts"]["embedded_chunks"] = len(embedded_chunks)

    console.print(f"Embedded {len(embedded_chunks)} chunks in {t1-t0:.2f} seconds", style="stage5")

    # --- STEP 6: Export (optional) ---
    if args.export and args.exporter:
        print_stage("STEP 6: Exporting", 5)
        t0 = time.time()

        exporter_type = args.exporter
        exporter = create_component_instance("exporter", exporter_type)
        export_path = exporter.export(embedded_chunks)

        t1 = time.time()
        pipeline_output["timings"]["exporting"] = t1 - t0
        pipeline_output["export_path"] = str(export_path)

        console.print(f"Exported documents to {export_path} in {t1-t0:.2f} seconds", style="stage6")

    # --- STEP 7: Vector store ---
    print_stage("STEP 7: Vector store", 6)
    t0 = time.time()

    vectorstore_type = args.vectorstore or settings.vectorstore
    vectorstore = create_component_instance(
        "vectorstore",
        vectorstore_type,
        collection_name=f"{args.customer}_parquet_test"
    )
    vectorstore.add(embedded_chunks)

    t1 = time.time()
    pipeline_output["timings"]["vectorstore"] = t1 - t0

    console.print(f"Added {len(embedded_chunks)} chunks to vector store in {t1-t0:.2f} seconds", style="stage7")

    # --- STEP 8: Retrieval ---
    print_stage("STEP 8: Retrieval", 7)
    t0 = time.time()

    # Example query
    query = "What is Coupa?"
    retriever_type = args.retriever or settings.retriever
    retriever = create_component_instance(
        "retriever",
        retriever_type,
        vectorstore=vectorstore
    )
    results = retriever.retrieve(query)

    t1 = time.time()
    pipeline_output["timings"]["retrieval"] = t1 - t0
    pipeline_output["counts"]["retrieval_results"] = len(results)

    console.print(f"Retrieved {len(results)} results for query '{query}' in {t1-t0:.2f} seconds", style="stage8")
    print_sample(results, n=1, stage_idx=7)

    # --- STEP 9: RAG Q&A (optional) ---
    default_questions = [
        "What is Coupa?",
        "How do I set up a workflow in Coupa?",
        "What are the key features of Coupa?"
    ]

    if args.run_qa:
        print_stage("STEP 9: RAG Q&A", 8)
        refresh_client()  # Reset any global LLM client state

        for i, question in enumerate(default_questions):
            t0 = time.time()
            console.print(f"\n[bold]Question {i+1}[/bold]: {question}")

            response = process_rag_query(
                query=question,
                retriever=retriever,
                max_tokens=1000
            )

            t1 = time.time()
            formatted = format_rag_response(response)

            console.print("[bold]Answer[/bold]:", style="stage9")
            console.print(formatted["answer"])
            console.print(f"[dim](Generated in {t1-t0:.2f} seconds)[/dim]")

            pipeline_output.setdefault("qa_results", []).append({
                "question": question,
                "answer": formatted["answer"],
                "sources": formatted.get("sources", []),
                "time": t1 - t0
            })

    # --- Summary ---
    total_time = time.time() - start_time
    pipeline_output["timings"]["total"] = total_time

    print_stage("PIPELINE SUMMARY", 8)
    console.print(f"Total processing time: {total_time:.2f} seconds")
    console.print(f"Documents: {len(raw_docs)} → Chunks: {len(chunks)}")

    for stage, timing in pipeline_output["timings"].items():
        if stage != "total":
            console.print(f"  {stage}: {timing:.2f}s ({timing/total_time*100:.1f}%)")

    # Save output to file
    with open(args.output, "w") as f:
        json.dump(pipeline_output, f, indent=2)

    console.print(f"\nPipeline diagnostics saved to {args.output}")

if __name__ == "__main__":
    main()
