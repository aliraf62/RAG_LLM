# Usage Examples: core.context_formatter

This guide demonstrates how to use the prompt construction and citation utilities.

## Example: Build a Context Prompt

```python
from core.rag.context_formatter import build_context_prompt

docs = [
    {
        "text": "This is a sample document about sourcing.",
        "url": "https://example.com/guide1",
        "images": ["images/guide1.png"],
        "caption": "Guide 1 Screenshot"
    },
    {
        "text": "Another document about procurement.",
        "url": "https://example.com/guide2"
    }
]

prompt = build_context_prompt(
    question="How do I create a sourcing event?",
    primary_domain="sourcing",
    docs=docs,
    include_images=True
)
print(prompt)
```

**Output:**
```
Domain: sourcing
Question: How do I create a sourcing event?

Context:
![Guide 1 Screenshot](images/guide1.png)
[Source](https://example.com/guide1)
This is a sample document about sourcing.

[Source](https://example.com/guide2)
Another document about procurement.
```

## Example: Get Citation Text

```python
from core.rag.context_formatter import get_citation_text

docs = [
    {"rating-name": "Guide 1"},
    {"guide-name": "Guide 2"},
    {"rating-name": "Guide 1"}  # Duplicate, will be deduplicated
]

citations = get_citation_text(docs)
print(citations)
```

**Output:**
```
**Cited guides:**
- Guide 1
- Guide 2
```
