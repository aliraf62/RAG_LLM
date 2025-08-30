# Coupa AI Assistant Configuration Guide

This guide explains the configuration options available for the Coupa AI Assistant, as defined in `core/defaults.py`. These settings control model selection, retrieval, formatting, chunking, and more. You can override these defaults using a YAML config file or environment variables.

---

## 1. Model Configuration

- **MODEL**: Name of the default language model (e.g., `"gpt-4o-mini"`).
- **VISION_MODEL**: Model used for vision tasks (e.g., `"gpt-4o-mini"`).
- **EMBED_MODEL**: Embedding model for vector search (e.g., `"text-embedding-3-small"`).
- **DEFAULT_TEMPERATURE**: Controls randomness in LLM output. `0.0` is deterministic, higher values increase creativity.

## 2. API and Request Configuration

- **SCOPE**: Project or organization scope (default: `"coupa"`).
- **USE_CASE**: Use case identifier (default: `"ai-indexing"`).
- **USE_PERSONAL_API**: If `True`, uses a personal API key (deprecated).

## 3. File Paths

- **VECTOR_STORE_PATH**: Path to the vector store directory.
- **HTML_EXPORT_DIR**: Directory for exported HTML files.

## 4. Retrieval Parameters

- **TOP_K**: Number of documents to retrieve per query.
- **SIMILARITY_THRESHOLD**: Minimum similarity score (0.0–1.0) for retrieved documents.

## 5. LLM Parameters

- **MAX_TOKENS**: Maximum tokens in LLM responses.
- **MAX_HISTORY_TOKENS**: Maximum tokens for conversation history.
- **TOKEN_COUNT_METHOD**: How to estimate token usage (`"word"` or `"char"`).

## 6. Output Formatting

- **ENABLE_CITATIONS**: If `True`, output includes citations.
- **DEDUPLICATE_SOURCES**: If `True`, deduplicates sources in output.

## 7. RAG (Retrieval-Augmented Generation) Prompts

- **RAG_SYSTEM_PROMPT**: System prompt for the LLM.
- **RAG_USER_TEMPLATE**: Template for user prompts, with `{context}` and `{question}` placeholders.
- **RAG_CONTEXT_FORMAT**: Format for displaying context passages.
- **RAG_RAW_OUTPUT_MAX_PREVIEW**: Max characters to show in debug previews.
- **RAG_SOURCES_HEADER**: Header for sources section.
- **RAG_SOURCE_FORMAT**: Format for listing sources.
- **PROMPT_TEMPLATES**: Domain-specific prompt templates (e.g., `"default"`, `"cso"`, `"sourcing"`).

## 8. Filtering

- **FILTER_LOW_SCORES**: If `True`, filters out low-scoring retrievals.

## 9. Image Handling

- **INCLUDE_IMAGES**: If `True`, includes images in output.

## 10. Chunking Parameters

- **DEFAULT_CHUNK_SIZE** / **DEFAULT_CHUNK_OVERLAP**: Chunk size and overlap for generic documents.
- **HTML_CHUNK_SIZE** / **HTML_CHUNK_OVERLAP**: Chunk size and overlap for HTML documents.
- **MARKDOWN_CHUNK_SIZE** / **MARKDOWN_CHUNK_OVERLAP**: Chunk size and overlap for Markdown documents.

## 11. Cleaning Parameters

- **MARKDOWN_CLEANER_PRESERVE_LINKS**: If `True`, preserves links in Markdown cleaning.
- **TEXT_CLEANER_MAX_LENGTH**: Maximum length for text cleaning.

## 12. HTML Cleaner Configuration

- **HTML_CLEANER_PARSER**: Parser to use (e.g., `"lxml"`).
- **HTML_CLEANER_REMOVE_TAGS**: List of HTML tags to remove.
- **HTML_CLEANER_PRESERVE_TABLES**: If `True`, preserves tables.
- **HTML_CLEANER_PRESERVE_HEADING_HIERARCHY**: If `True`, preserves heading levels.
- **HTML_CLEANER_PRESERVE_LISTS**: If `True`, preserves list structure.
- **HTML_CLEANER_EXTRACT_METADATA**: If `True`, extracts metadata.
- **HTML_CLEANER_PRESERVE_LINKS**: If `True`, preserves links.
- **HTML_CLEANER_PRESERVE_IMAGES**: If `True`, preserves images.
- **HTML_CLEANER_OUTPUT_FORMAT**: Output format (e.g., `"text"`).

## 13. Text Cleaner Configuration

- **TEXT_CLEANER_NORMALIZE_WHITESPACE**: If `True`, normalizes whitespace.
- **TEXT_CLEANER_NORMALIZE_NEWLINES**: If `True`, normalizes newlines.
- **TEXT_CLEANER_SENTENCE_SEGMENTATION**: If `True`, segments sentences.
- **TEXT_CLEANER_EXTRACT_METADATA**: If `True`, extracts metadata.
- **TEXT_CLEANER_OUTPUT_FORMAT**: Output format (e.g., `"text"`).

## 14. Markdown Cleaner Configuration

- **MARKDOWN_CLEANER_PRESERVE_HEADING_HIERARCHY**: If `True`, preserves heading levels.
- **MARKDOWN_CLEANER_PRESERVE_LIST_STRUCTURE**: If `True`, preserves list structure.
- **MARKDOWN_CLEANER_EXTRACT_METADATA**: If `True`, extracts metadata.
- **MARKDOWN_CLEANER_OUTPUT_FORMAT**: Output format (e.g., `"text"`).
- **MARKDOWN_CLEANER_EXTRACT_FRONTMATTER**: If `True`, extracts YAML frontmatter.

---

## How to Override Defaults

You can override these defaults by:
- Editing your `config.yaml` file.
- Setting environment variables (e.g., `export MODEL="gpt-4"`).

For more details, see `docs/architecture/defaults.md`.

---

## Example

```yaml
MODEL: "gpt-4"
TOP_K: 10
ENABLE_CITATIONS: false
```

---

## Further Reading

- [Architecture Overview](../architecture/defaults.md)
- [Cleaning and Chunking Strategies](../architecture/cleaning.md)
- [RAG Prompt Engineering](../architecture/rag.md)