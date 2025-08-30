# Example: Customizing Coupa AI Assistant Configuration

This example demonstrates how to override the default configuration for the Coupa AI Assistant using a YAML file and environment variables.

---

## 1. Overriding with `config.yaml`

Suppose you want to use a different model, increase the number of retrieved documents, and disable citations.  
Create or edit your `config.yaml`:

```yaml
MODEL: "gpt-4"
TOP_K: 10
ENABLE_CITATIONS: false
SIMILARITY_THRESHOLD: 0.3
VECTOR_STORE_PATH: "vector_store/my_custom_index"
```

The assistant will now use the GPT-4 model, retrieve 10 documents per query, and omit citations in answers.

---

## 2. Overriding with Environment Variables

You can also override settings at runtime:

```bash
export MODEL="gpt-4"
export TOP_K=10
export ENABLE_CITATIONS=false
```

Environment variables take precedence over defaults and YAML config.

---

## 3. Using the Configuration in Code

You can access configuration values in your Python code via the `settings` object:

```python
from core.settings import settings

model = settings.get("MODEL")
top_k = settings.get("TOP_K", 5)
enable_citations = settings.get("ENABLE_CITATIONS", True)

print(f"Model: {model}, Top K: {top_k}, Citations: {enable_citations}")
```

---

## 4. Example Query with Custom settings

```python
from core.rag_processor import process_rag_query, format_rag_response

result = process_rag_query(
    question="How do I reset my password?",
    top_k=10,  # This will override the settings value for this call
    raw=False
)
print(format_rag_response(result))
```

---

## 5. Tips

- Place your `config.yaml` in the project root or specify its path via an environment variable if needed.
- For a full list of options, see [../guides/ai_assistant_config_guide.md](../guides/ai_assistant_config_guide.md).