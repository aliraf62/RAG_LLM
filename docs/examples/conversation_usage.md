# Usage Examples: core.conversation

This guide demonstrates how to use the conversation management and chat-completion utilities.

## Example: Trim Conversation History

```python
from core.conversation import trim_conversation_history

history = [
    {"role": "user", "content": "Hello, how do I create an event?"},
    {"role": "assistant", "content": "To create an event, go to..."},
    {"role": "user", "content": "What about adding suppliers?"},
    {"role": "assistant", "content": "You can add suppliers by..."},
    # ... more messages ...
]

trimmed = trim_conversation_history(history)
print(trimmed)
```

## Example: Chat Completion

```python
from core.conversation import chat_completion

system_prompt = "You are a helpful assistant."
user_prompt = "How do I reset my password?"
history = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"}
]

response = chat_completion(
    system=system_prompt,
    user=user_prompt,
    history=history,
    max_tokens=128,
    temperature=0.2
)
print(response)
```
