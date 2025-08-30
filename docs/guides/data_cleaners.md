# Guide: Data Cleaners

This guide explains the architecture, usage, and extension of content cleaners in the `data_ingestion/cleaners/` module. Cleaners are responsible for transforming raw HTML, Markdown, or plain text into normalized, metadata-enriched text suitable for embedding and retrieval in RAG (Retrieval-Augmented Generation) pipelines.

---

## 1. Purpose

Cleaners standardize and enrich content from diverse sources, ensuring that downstream chunking, embedding, and retrieval operate on high-quality, semantically meaningful text.

---

## 2. Base Architecture

All cleaners inherit from `BaseCleaner` (see `cleaner_base.py`), which provides:

- **Validation**: Ensures content is non-empty and of the correct type.
- **Configuration**: Uses `get_config_value()` for consistent config retrieval.
- **Template Method**: The `clean_for_rag()` method wraps cleaning, metadata extraction, and enhancement.
- **Extensibility**: Subclasses can override `enhance_rag_result()` to add custom fields.

### Example: BaseCleaner

```python
class BaseCleaner(ABC):
    def clean_for_rag(self, content: str, **kwargs) -> Dict[str, Any]:
        self.validate_content(content)
        # ... get config, clean, extract metadata ...
        result = {
            "text": cleaned_text,
            "format": output_format
        }
        if extract_metadata:
            result["metadata"] = self.extract_metadata(content)
        self.enhance_rag_result(content, result, **kwargs)
        return result
```

---

## 3. Cleaner Types

- **HTMLCleaner**: Cleans HTML, removes boilerplate, preserves structure, extracts metadata, and can convert to Markdown.
- **MarkdownCleaner**: Cleans Markdown, preserves headings/lists, extracts YAML frontmatter, and can convert to HTML.
- **TextCleaner**: Cleans plain text, normalizes whitespace, segments sentences, and extracts simple metadata.

Each cleaner implements:
- `clean()`
- `extract_metadata()`
- Optionally: `enhance_rag_result()`, `to_html()`, `to_markdown()`

---

## 4. Configuration

Cleaners use config keys (e.g., `HTML_CLEANER_PRESERVE_TABLES`) and allow overrides via kwargs.  
You can set defaults in your config or pass options at runtime.

---

## 5. Usage Example

```python
from core.pipeline import clean_content

raw_html = "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
result = clean_content(raw_html)
print(result["text"])
print(result["metadata"])
```

---

## 6. Metadata Extraction

- **HTMLCleaner**: Extracts `<title>`, `<meta>`, and JSON-LD.
- **MarkdownCleaner**: Extracts YAML frontmatter and first heading as title.
- **TextCleaner**: Extracts title, author, date, and subject from first lines.

---

## 7. Extending Cleaners

To add a new cleaner:

1. Subclass `BaseCleaner`.
2. Implement `clean()` and `extract_metadata()`.
3. Register in the `CLEANER_REGISTRY` in `__init__.py` if needed.
4. Optionally override `enhance_rag_result()` for custom output.

---

## 8. References

- [data_ingestion/cleaners/cleaner_base.py](../../core/pipeline/cleaners/base.py)
- [data_ingestion/cleaners/html_cleaner.py](../../core/pipeline/cleaners/html_cleaner.py)
- [data_ingestion/cleaners/markdown_cleaner.py](../../core/pipeline/cleaners/markdown_cleaner.py)
- [data_ingestion/cleaners/text_cleaner.py](../../core/pipeline/cleaners/text_cleaner.py)
- [data_ingestion/cleaners/__init__.py](../../core/pipeline/cleaners/__init__.py)

---
