"""
Example: Using TextChunker

Demonstrates how to use the TextChunker class to split plain text into token-aware, heading-aware chunks.
"""

from data_ingestion.chunkers.text_chunker import TextChunker

text_content = """
# Introduction

This is the introduction section. It is short.

## Details

Here are some details about the topic. This section is intentionally made longer to demonstrate chunking.
""" + " ".join(["word"] * 1000) + """

## Conclusion

Summary and final thoughts.
"""

metadata = {"doc_id": "txt_example_1"}

chunker = TextChunker()
chunks = chunker.chunk(content=text_content, metadata=metadata)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print("Type:", chunk.metadata.get("chunk_type"))
    print("Token count:", chunk.metadata.get("token_count"))
    print("Content:", chunk.page_content[:80], "...")
    print("---")
