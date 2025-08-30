#!/usr/bin/env python
"""
Direct FAISS index builder for the AI QnA Assistant.

This script directly builds a FAISS vector index without relying on complex abstractions.
It ensures index files are properly created for reliable vector search.

Example usage:
    python scripts/direct_index_builder.py --input temp/extracted_docs.json --output vector_store/direct_test
"""

import argparse
import json
import sys
import os
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
from langchain.schema import Document

# Add parent directory to path to import from the root package
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.llm import get_llm_client

def load_documents(input_path: str) -> List[Document]:
    """Load documents from a JSON file"""
    print(f"Loading documents from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for item in data:
        if "text" in item and "metadata" in item:
            doc = Document(
                page_content=item["text"],
                metadata=item["metadata"]
            )
            documents.append(doc)

    print(f"Loaded {len(documents)} documents")
    return documents

def build_index(
    docs: List[Document],
    output_dir: str,
    embedding_dimension: int = 1536,
    batch_size: int = 64
) -> None:
    """Build a FAISS index directly without relying on complex abstractions"""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    index_path = output_path / "index.faiss"
    metadata_path = output_path / "metadata.json"

    print(f"Building index in {output_dir}")
    print(f"FAISS index will be saved to {index_path}")
    print(f"Metadata will be saved to {metadata_path}")

    # Get embedding function from OpenAI
    client = get_llm_client()

    # Extract text and prepare metadata
    texts = [doc.page_content for doc in docs]
    metadata = [doc.metadata for doc in docs]

    print(f"Computing embeddings for {len(texts)} documents (in batches of {batch_size})...")

    # Compute embeddings in batches
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:min(i + batch_size, len(texts))]
        batch_embeddings = client.get_embeddings(batch)
        embeddings.extend(batch_embeddings)
        print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

    # Convert embeddings to numpy array
    vectors = np.array(embeddings, dtype=np.float32)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)

    # Build FAISS index
    index = faiss.IndexFlatIP(embedding_dimension)  # Inner product for cosine similarity
    index.add(vectors)

    # Save index and metadata
    print(f"Saving index to {index_path}")
    faiss.write_index(index, str(index_path))

    # Save metadata along with original texts
    metadata_objects = []
    for i, (meta, text) in enumerate(zip(metadata, texts)):
        metadata_objects.append({
            **meta,
            "text": text,
            "id": i
        })

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_objects, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n✅ Index built successfully with {len(texts)} documents")
    print(f"   Index file: {index_path} ({index_path.stat().st_size / 1024:.2f} KB)")
    print(f"   Metadata file: {metadata_path} ({metadata_path.stat().st_size / 1024:.2f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index directly")
    parser.add_argument("--input", required=True,
                       help="JSON file with documents to index (output from extract step)")
    parser.add_argument("--output", required=True,
                       help="Output directory for vector store (will be created if it doesn't exist)")
    parser.add_argument("--dimension", type=int, default=1536,
                       help="Embedding dimension (default: 1536)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for embedding computation (default: 64)")

    args = parser.parse_args()

    # Load documents
    docs = load_documents(args.input)
    if not docs:
        print("Error: No documents loaded")
        return 1

    # Build index
    build_index(
        docs=docs,
        output_dir=args.output,
        embedding_dimension=args.dimension,
        batch_size=args.batch_size
    )

    return 0

if __name__ == "__main__":
    sys.exit(main())
