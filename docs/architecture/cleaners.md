# Cleaners Architecture

## Design Principles

- **Pluggable**: Each cleaner is a class implementing a common interface (`BaseCleaner`).
- **Unified API**: All cleaners support `clean`, `extract_metadata`, and `clean_for_rag`.
- **Extensible**: New content types can be supported by subclassing `BaseCleaner`.
- **Configurable**: Cleaners accept configuration via kwargs or environment/config.

## Class Hierarchy

- `BaseCleaner` (abstract)
  - `HTMLCleaner`
  - `MarkdownCleaner`
  - `TextCleaner`

## Key Methods

- `clean(content, **kwargs)`: Main cleaning logic.
- `extract_metadata(content)`: Extracts metadata (title, author, etc).
- `clean_for_rag(content, **kwargs)`: Returns a dict with cleaned text, metadata, and format.
- `enhance_rag_result(content, result, **kwargs)`: Optional hook for adding extra outputs.

## Content Detection

The `detect_content_type` function uses regex patterns to auto-detect content type.

## Example Flow

1. User calls `clean_content(raw_content)`.
2. Content type is detected.
3. Appropriate cleaner is instantiated.
4. Cleaner processes content and returns a result dict.

## References

- See [docs/guides/cleaners.md](../guides/cleaners.md) for usage.
- See [docs/examples/](../examples/) for code samples.