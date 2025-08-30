# Guide: TextChunker

This guide explains the usage and design of the `TextChunker` class in `data_ingestion/chunkers/text_chunker.py`. The `TextChunker` is responsible for splitting plain text documents into token-aware, heading-aware chunks for downstream embedding and retrieval.

---

## Purpose

- **Token-aware chunking**: Ensures each chunk fits within a target token budget, using `tiktoken` if available.
- **Heading-aware chunking**: Splits text at semantic boundaries such as Markdown-style headings or bullets.
- **Overlap**: Supports overlapping chunks to preserve context between segments.

---

## How It Works

1. **Heading Splitting**:  
   The text is first split on Markdown-style headings (e.g., `#`, `##`, `-`, etc.) to preserve semantic blocks.
2. **Token Counting**:  
   Each block is further split into chunks of up to `max_tokens` tokens (default: 800), with `overlap` tokens (default: 100) repeated between chunks.
   - Uses `tiktoken` for accurate token counting if available, otherwise falls back to an approximate method.
3. **Chunk Metadata**:  
   Each chunk includes metadata such as `chunk_index`, `token_count`, and word indices.

---

## Usage Example

See [../examples/text_chunker.py](../examples/text_chunker.py):

```python
from core.pipeline.chunkers import TextChunker

text_content = "# Heading\n\n" + "word " * 1000
metadata = {"doc_id": "txt_example_1"}

chunker = TextChunker()
chunks = chunker.chunk(content=text_content, metadata=metadata)

for i, chunk in enumerate(chunks):
   print(f"Chunk {i}:")
   print("Type:", chunk.metadata.get("chunk_type"))
   print("Token count:", chunk.metadata.get("token_count"))
   print("Content:", chunk.page_content[:80], "...")
   print("---")
```

---

## Configuration

- `DEFAULT_CHUNK_SIZE`: Default max tokens per chunk (configurable).
- `DEFAULT_CHUNK_OVERLAP`: Default overlap between chunks (configurable).

You can override these via the config or by passing arguments to `chunk()`.

---

## References

- [data_ingestion/chunkers/text_chunker.py](../../core/pipeline/chunkers/text_chunker.py)
- [Guide: Data Chunkers](data_chunkers.md)
