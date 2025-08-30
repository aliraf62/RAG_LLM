#!/usr/bin/env python3
"""
Build or update a FAISS index from exported HTML guides.

This script processes exported HTML files, chunks them according to the specified strategy,
embeds the content, and builds a FAISS index for efficient retrieval.
See docs/architecture/vector_indexing.md for design rationale.
"""
from __future__ import annotations

import argparse, sys, logging
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#  Make sure repo‑root is on PYTHONPATH when invoked as a script
# --------------------------------------------------------------------------- #
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

# ---- local imports -------------------------------------------------------- #
from core.pipeline import HTMLFSLoader
from core.pipeline.chunkers import HTMLChunker
from core.pipeline.indexing.indexer import build_faiss_index
from core.async_embed import embed_batch
from core.config import refresh_openai_client

# Set up logger
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
def _chunk_stream(loader: Iterable, strategy: str):
    """
    Yield LangChain `Document`s according to the chosen strategy.

    Args:
        loader (Iterable): Iterable of LangChain Documents.
        strategy (str): Chunking strategy ("none" or "header").

    Yields:
        LangChain Document objects, chunked as specified.

    See docs/examples/chunk_stream.md for usage examples.
    """
    chunker = HTMLChunker()
    for doc in loader:
        if strategy == "none":
            yield doc
        else:
            for c in chunker.chunk(content=doc.page_content, metadata=doc.metadata):
                yield c


def build_index_from_html(html_dir, out_dir, chunk_strategy=None, limit=None, batch_size=64):
    """
    Build a FAISS index from HTML files.

    Args:
        html_dir: Directory containing HTML files
        out_dir: Output directory for FAISS index
        chunk_strategy: Chunking strategy ("none" or "header")
        limit: Maximum number of files to process (None for all)
        batch_size: Batch size for embeddings

    Returns:
        Path to the output directory
    """
    html_path = Path(html_dir)

    try:
        # The HTMLFSLoader now uses cleaners internally
        loader = HTMLFSLoader(html_path)
        logger.info(f"Initialized HTML loader for directory: {html_path}")

        # Apply chunking strategy
        stream = _chunk_stream(loader, chunk_strategy)

        # Apply limit if specified
        if limit:
            logger.info(f"Limiting processing to {limit} documents")
            stream = (d for i, d in enumerate(stream) if i < limit)

        # Build the index
        logger.info(f"Building FAISS index in {out_dir} with batch size {batch_size}")
        build_faiss_index(
            docs=tqdm(stream, desc="Processing", ncols=80),
            faiss_dir=out_dir,
            embed_fn=embed_batch,
            batch_size=batch_size,
        )

        return out_dir
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise


def main() -> None:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    ap = argparse.ArgumentParser(description="Build / update FAISS from exported HTML")
    ap.add_argument("--html-dir", type=Path, required=True,
                    help="Directory that contains *.html plus images/ & files/")
    ap.add_argument("--out", type=Path, required=True,
                    help="Folder where index.faiss + metadata.jsonl will be written")
    ap.add_argument("--chunk", choices=("none", "header"), default="header",
                    help="Chunking strategy (default: header)")
    ap.add_argument("--batch", type=int, default=64,
                    help="Embedding batch size (default: 64)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: process only N docs (0 = all)")
    args = ap.parse_args()

    html_path = args.html_dir.expanduser().resolve()
    if not html_path.exists():
        sys.exit(f"❌  HTML directory not found: {html_path}")

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Refresh token before starting embeddings
    refresh_openai_client()

    print("📦  Embedding & indexing …")
    try:
        result = build_index_from_html(
            html_dir=html_path,
            out_dir=out_dir,
            chunk_strategy=args.chunk,
            limit=None if args.limit == 0 else args.limit,
            batch_size=args.batch,
        )
        print(f"✅  FAISS ready at {result}")
    except Exception as e:
        print(f"❌  Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()