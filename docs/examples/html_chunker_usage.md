# Example: Using HTMLChunker

This example demonstrates how to use the `HTMLChunker` class to split HTML content into semantically meaningful chunks for downstream embedding and retrieval.

---

## Basic Usage

```python
from core.pipeline.chunkers import HTMLChunker

html_content = """
<html>
  <body>
    <h1>Introduction</h1>
    <p>This is the intro section.</p>
    <h2>Details</h2>
    <p>More details here.</p>
    <figure>
      <img src="image.png" />
      <figcaption>Example image</figcaption>
    </figure>
    <h2>Conclusion</h2>
    <p>Summary text.</p>
  </body>
</html>
"""

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

## Output

- Each `<figure>` is extracted as a separate chunk with `chunk_type: "figure"`.
- Other content is split by headers (`h1`, `h2`, etc.), and further split by character count if needed.
- Each chunk includes metadata such as `chunk_type` and (optionally) `chunk_index`.

---

## See Also

- [Guide: Data Chunkers](../guides/data_chunkers.md)
