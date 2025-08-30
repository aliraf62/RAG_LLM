# Guide: Data Chunkers in ai_qna_assistant

This guide explains the architecture, usage, and extension of chunkers in the `data_ingestion/chunkers/` module. Chunkers are responsible for splitting documents into smaller, semantically meaningful pieces for embedding and retrieval.

---

## 1. Purpose

Chunkers break up large documents (HTML, Markdown, plain text, etc.) into smaller "chunks" or passages. This improves retrieval granularity and relevance in RAG (Retrieval-Augmented Generation) pipelines.

---

## 2. Base Architecture

All chunkers inherit from `BaseChunker` (see `chunker_base.py`), which provides:

- **Validation**: Ensures content and metadata are well-formed.
- **Template Method**: The `chunk()` method wraps validation and error handling around the subclass's `_perform_chunking()` implementation.
- **Error Handling**: If chunking fails, an error document is returned.
- **Configuration**: Uses `ConfigurationMixin` for consistent config value retrieval.

### Example: BaseChunker

```python
class BaseChunker(ABC, ConfigurationMixin):
    def chunk(self, content: str, metadata: Dict[str, Any], **kwargs) -> List[Document]:
        self.validate_content(content)
        self.validate_metadata(metadata)
        try:
            return self._perform_chunking(content, metadata, **kwargs)
        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}")
            return [self._create_error_document(content, metadata, str(e))]
```

---

## 3. Chunker Types

### HTMLChunker

- Splits HTML documents by header tags (`h1`, `h2`, `h3`) to preserve structure.
- Extracts `<figure>` elements as standalone chunks.
- Further splits long sections by character count.
- See [html_chunker.py](../../core/pipeline/chunkers/html_chunker.py) and [examples/html_chunker_usage.md](../examples/html_chunker_usage.md).

### MarkdownChunker

- Splits Markdown into overlapping, token-aware chunks.
- Supports configurable chunk size, overlap, and custom tokenizers.
- Preserves semantic structure (headings, lists, code blocks).
- See [markdown_chunker.py](../../core/pipeline/chunkers/markdown_chunker.py) and [examples/markdown_chunker_usage.md](../examples/markdown_chunker_usage.md).

### TextChunker

- Splits plain text by headings and token count.
- Uses `tiktoken` for accurate token counting if available.
- Supports overlap and semantic block preservation.
- See [text_chunker.py](../../core/pipeline/chunkers/text_chunker.py) and [examples/text_chunker.md](../examples/text_chunker.md).

---

## 4. Configuration

Chunkers use the `ConfigurationMixin` to fetch settings from the global config, allowing for runtime customization (e.g., chunk size, overlap).

```python
chunk_size = self.get_config_value(
    param_value=kwargs.get("chunk_size"),
    config_key="HTML_CHUNK_SIZE",
    default_value=500
)
```

---

## 5. Usage Examples

- [HTMLChunker Example](../examples/html_chunker_usage.md)
- [MarkdownChunker Example](../examples/markdown_chunker_usage.md)
- [TextChunker Example](../examples/text_chunker.py)

---

## 6. Backward Compatibility

The `BackwardCompatibilityMixin` provides a `get_instance()` method for singleton-like usage, supporting legacy code that expects a default instance.

---

## 7. Extending Chunkers

To add a new chunker:

1. Subclass `BaseChunker`.
2. Implement `_perform_chunking()` and `_get_chunk_type()`.
3. Register or use as needed.

---

## 8. References

- [data_ingestion/chunkers/chunker_base.py](../../core/pipeline/chunkers/base.py)
- [data_ingestion/chunkers/html_chunker.py](../../core/pipeline/chunkers/html_chunker.py)
- [data_ingestion/chunkers/markdown_chunker.py](../../core/pipeline/chunkers/markdown_chunker.py)
- [data_ingestion/chunkers/text_chunker.py](../../core/pipeline/chunkers/text_chunker.py)

---
````