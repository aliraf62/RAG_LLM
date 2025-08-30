# TextCleaner Example

```python
from core.pipeline import TextCleaner

text = """
Title: Plain Text Example
Author: Carol

This is a plain text document.
It has multiple lines.

The end.
"""

cleaner = TextCleaner()
result = cleaner.clean_for_rag(text)

print("Cleaned text:")
print(result["text"])
print("Metadata:")
print(result["metadata"])
print("Sentences:")
print(result["sentences"])
```

**Output:**
```
Cleaned text:
Title: Plain Text Example
Author: Carol

This is a plain text document.
It has multiple lines.

The end.

Metadata:
{'title': 'Plain Text Example', 'author': 'Carol'}

Sentences:
['Title: Plain Text Example', 'Author: Carol', 'This is a plain text document.', 'It has multiple lines.', 'The end.']
```