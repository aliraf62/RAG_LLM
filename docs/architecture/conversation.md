# Conversation Management Architecture

## Overview

The `core.conversation` module provides utilities for managing conversation history and enforcing token budgets in chat-based LLM applications. It also wraps the OpenAI chat-completion API with history trimming.

## Design Rationale

- **Token Budget Enforcement:** Ensures that the conversation history does not exceed the configured token limit, preventing API errors and optimizing context.
- **Configurable Token Counting:** Supports both word-based and character-based token estimation, configurable via `TOKEN_COUNT_METHOD`.
- **Separation of Concerns:** Keeps conversation management logic separate from RAG and retrieval logic.

## Key Functions

- `trim_conversation_history`: Trims the oldest exchanges from history to fit within the token budget.
- `chat_completion`: Sends a chat-completion request to the OpenAI API, automatically trimming history as needed.

## Example Usage

See [docs/examples/conversation_usage.md](../examples/conversation_usage.md).

## See Also

- [core/conversation.py](../../core/conversation.py)
- [Configuration system](config.md)
