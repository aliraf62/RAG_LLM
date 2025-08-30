# Example: Using MarkdownChunker

This example demonstrates how to use the `MarkdownChunker` class and the `split_markdown` function to split Markdown content into overlapping, token-aware chunks for embedding and retrieval.

---

## Basic Usage with MarkdownChunker

```python
from core.pipeline.chunkers import MarkdownChunker

markdown_content = """
# Introduction

This is the introduction section.

## Details

Here are some details about the topic.

- Bullet point one
- Bullet point two

## Conclusion

Summary and final thoughts.
"""

metadata = {"doc_id": "md_example_1"}

chunker = MarkdownChunker()
chunks = chunker.chunk(content=markdown_content, metadata=metadata)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print("Type:", chunk.metadata.get("chunk_type"))
    print("Content:", chunk.page_content[:80], "...")
    print("---")
```

---

## Using split_markdown Directly

```python
from core.pipeline.chunkers import split_markdown

markdown_content = "# Heading\n\n" + "word " * 350  # Simulate a long section

chunks = split_markdown(markdown_content, max_tokens=100, overlap=20)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i} ({len(chunk.split())} words): {chunk[:60]}...")
```

---

## Output

- Chunks are created with overlap to preserve context.
- Each chunk is suitable for embedding or retrieval in a RAG pipeline.

---

## Advanced: Custom Tokenizer

You can provide your own tokenizer for special tokenization needs:

```python
from core.pipeline.chunkers import split_markdown


def custom_tokenizer(text):
    # Example: split by whitespace only
    return text.split()


markdown_content = "# Heading\n\n" + "word " * 350
chunks = split_markdown(markdown_content, max_tokens=50, overlap=10, tokenizer=custom_tokenizer)
print(f"Total chunks: {len(chunks)}")
```

---

## See Also

- [Guide: Data Chunkers](../guides/data_chunkers.md)
