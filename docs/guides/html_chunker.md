# Guide: HTMLChunker

This guide explains the usage and design of the `HTMLChunker` class in `data_ingestion/chunkers/html_chunker.py`. The `HTMLChunker` is responsible for splitting HTML documents into semantically meaningful chunks for embedding and retrieval.

---

## Purpose

- **Semantic chunking**: Splits HTML by header tags (`h1`, `h2`, `h3`) to preserve document structure.
- **Figure extraction**: Extracts each `<figure>` as a standalone chunk for better image handling.
- **Token-aware splitting**: Further splits long sections by character count, with configurable chunk size and overlap.
- **Robustness**: Handles malformed HTML gracefully and falls back to plain content splitting if header-based splitting fails.

---

## How It Works

1. **Figure Extraction**:  
   All `<figure>` elements are extracted as separate chunks, with their HTML and caption.
2. **Header Splitting**:  
   The remaining HTML is split by headers (`h1`, `h2`, `h3`) using `HTMLHeaderTextSplitter`.
3. **Recursive Splitting**:  
   Each header-based section is further split by character count if it exceeds the configured chunk size.
4. **Metadata**:  
   Each chunk includes metadata such as `chunk_type`, `chunk_index`, and (for figures) `caption`.

---

## Usage Example

See [../examples/html_chunker_usage.md](../examples/html_chunker_usage.md):

```python
from core.pipeline.chunkers import HTMLChunker

html_content = "<html>...</html>"
metadata = {"doc_id": "example_1"}

chunker = HTMLChunker()
chunks = chunker.chunk(content=html_content, metadata=metadata)

for i, chunk in enumerate(chunks):
   print(f"Chunk {i}:")
   print("Type:", chunk.metadata.get("chunk_type"))
   print("Content:", chunk.page_content[:80], "...")
   print("---")
```

---

## Configuration

- `HTML_CHUNK_SIZE`: Default max characters per chunk (configurable).
- `HTML_CHUNK_OVERLAP`: Default overlap between chunks (configurable).

You can override these via the config or by passing arguments to `chunk()`.

---

## Output

- `<figure>` elements become chunks with `chunk_type: "figure"`.
- Other content is split by headers and further by character count, with `chunk_type: "html"`.
- Each chunk includes relevant metadata.

---

## References

- [data_ingestion/chunkers/html_chunker.py](../../core/pipeline/chunkers/html_chunker.py)
- [Guide: Data Chunkers](data_chunkers.md)
