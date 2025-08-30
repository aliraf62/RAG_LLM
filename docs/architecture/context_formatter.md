# Context Formatter Architecture

## Overview

The `core.context_formatter` module provides utilities for constructing prompts and citation text from retrieved document metadata and content. It is used to build the context passed to LLMs and to generate citation footers for answers.

## Design Rationale

- **Separation of Formatting Logic:** Keeps prompt/citation formatting out of core retrieval and RAG logic.
- **Configurability:** Honors config flags such as `INCLUDE_IMAGES` and `ENABLE_CITATIONS`.
- **Extensibility:** Can be adapted to new document fields or output formats as needed.

## Key Functions

- `build_context_prompt`: Formats a prompt string from question, domain, and a list of document dicts.
- `get_citation_text`: Produces a citation footer listing cited guides, deduplicated.

## Example Usage

See [docs/examples/context_formatter_usage.md](../examples/context_formatter_usage.md).

## See Also

- [core/context_formatter.py](../../core/rag/context_formatter.py)
- [Configuration system](config.md)
