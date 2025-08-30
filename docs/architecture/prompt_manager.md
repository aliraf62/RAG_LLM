# Prompt Manager Architecture

## Overview

The `PromptManager` class provides a centralized, configurable way to manage system prompts that define the AI's persona and behavior. It selects the appropriate prompt template based on the dataset (domain) and question type, supporting flexible persona switching and context-aware responses.

## Design Rationale

- **Separation of Concerns:** Keeps prompt selection logic out of core business logic and model code.
- **Configurability:** Loads prompt templates from the configuration system, allowing easy updates and environment-specific overrides.
- **Extensibility:** Supports dataset-specific and question-type-specific prompts, with sensible fallbacks.

## Selection Logic

1. **Dataset + Question Type:** If a template exists for the combination (e.g., `"cso_howto"`), it is used.
2. **Dataset Only:** If a dataset-specific template exists (e.g., `"cso"`), it is used.
3. **Default:** Falls back to the `"default"` template if no specific match is found.

## Example Usage

```python
from core.rag.prompt_manager import prompt_manager

prompt = prompt_manager.get_system_prompt("cso", "howto")
```

## Configuration

Prompt templates are loaded from the `PROMPT_TEMPLATES` key in the config system (YAML, .env, or defaults).

## See Also

- [core/prompt_manager.py](../../core/rag/prompt_manager.py)
- [Configuration system](config.md)
