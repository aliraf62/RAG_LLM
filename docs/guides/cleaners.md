# Data Cleaners Guide

## Overview

Data cleaners are responsible for transforming raw content (HTML, Markdown, plain text) into a normalized, structured format suitable for Retrieval-Augmented Generation (RAG) and other downstream NLP tasks.

This guide covers:
- The role of each cleaner
- How to use them
- Configuration options

## Cleaner Types

- **HTMLCleaner**: Cleans and normalizes HTML documents, removes boilerplate, preserves structure (tables, lists, headings).
- **MarkdownCleaner**: Cleans Markdown, extracts frontmatter, preserves structure, can convert to HTML.
- **TextCleaner**: Cleans plain text, normalizes whitespace, extracts metadata, segments sentences.

## Usage

You can use the unified interface:

```python
from core.pipeline import clean_content

result = clean_content(raw_content)
print(result["text"])
print(result["metadata"])
```

Or use a specific cleaner:

```python
from core.pipeline import HTMLCleaner

cleaner = HTMLCleaner()
cleaned = cleaner.clean(raw_html)
```

## Configuration

You can override cleaner options via keyword arguments:

```python
result = clean_content(raw_content, output_format="markdown", preserve_tables=False)
```

Common options:
- `output_format`: "text", "markdown", or "html"
- `preserve_tables`, `preserve_lists`, `preserve_heading_hierarchy`
- `extract_metadata`, `extract_frontmatter`
- `normalize_whitespace`, `normalize_newlines`

## Extending

To add a new cleaner, subclass `BaseCleaner` and implement the required methods.

See also:
- [docs/architecture/cleaners.md](../architecture/cleaners.md)
- [docs/examples/htmls_cleaner.md](../examples/htmls_cleaner.md)
- [docs/examples/markdwn_cleaner.md](../examples/markdwn_cleaner.md)
- [docs/examples/text_cleaner.md](../examples/text_cleaner.md)