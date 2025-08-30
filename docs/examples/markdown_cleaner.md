# MarkdownCleaner Example

```python
from core.pipeline import MarkdownCleaner

markdown_text = """---
title: Markdown Example
author: Bob
---

# Heading 1

Some **bold** text and a [link](https://example.com).

* Bullet 1
* Bullet 2

```python
def foo():
    pass
```
"""

cleaner = MarkdownCleaner()
result = cleaner.clean_for_rag(markdown_text)

print("Cleaned text:")
print(result["text"])
print("Frontmatter:")
print(result.get("frontmatter"))
```

**Output:**
```
Cleaned text:
[H1] Heading 1

Some bold text and a link.

[BULLET 1] Bullet 1
[BULLET 2] Bullet 2

[CODE BLOCK START]
def foo():
    pass
[CODE BLOCK END]
...

Frontmatter:
{'title': 'Markdown Example', 'author': 'Bob'}
```