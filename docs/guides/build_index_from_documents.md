# Guide: Building a FAISS Index from Exported HTML

This guide explains how to use `scripts/build_index_from_html.py` to build or update a FAISS vector index from a directory of exported HTML guides.

## Overview

The script processes HTML files, optionally chunks them, generates embeddings, and writes a FAISS index and metadata for retrieval-augmented generation (RAG) pipelines.

## Usage

```bash
python scripts/build_index_from_html.py \
    --html-dir <HTML_EXPORT_DIR> \
    --out      <OUTPUT_INDEX_DIR> \
    --chunk    <none|header> \
    --batch    <BATCH_SIZE> \
    --limit    <N>
```

- `--html-dir`: Path to directory containing exported HTML files (required).
- `--out`: Output directory for the FAISS index and metadata (required).
- `--chunk`: Chunking strategy:  
  - `none` (no chunking, each HTML file is a single document)  
  - `header` (split by headers, default)
- `--batch`: Embedding batch size (default: 64).
- `--limit`: Process only the first N documents (0 = all, default: 0).

## Example

```bash
python scripts/build_index_from_documents.py \
    --html-dir outputs/CSO_workflow_html_exports/html \
    --out      vector_store/cso_index \
    --chunk    header \
    --batch    64 \
    --limit    0
```

## Notes

- The script expects HTML files exported in a compatible format (see `data_ingestion/exporters/cso_html_exporter.py`).
- The output directory will contain `index.faiss` and `metadata.jsonl`.
- For more details on the architecture, see [docs/architecture/vector_indexing.md](../architecture/vector_indexing.md).

## Troubleshooting

- Ensure the HTML directory exists and contains valid files.
- If you encounter authentication errors, check your OpenAI API credentials.
