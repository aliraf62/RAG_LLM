# HTMLCleaner Example

```python
from core.pipeline import HTMLCleaner

html = """
<html>
  <head>
    <title>Sample HTML</title>
    <meta name="author" content="Alice">
  </head>
  <body>
    <h1>Welcome</h1>
    <p>This is a <b>test</b> document.</p>
    <ul>
      <li>Item 1</li>
      <li>Item 2</li>
    </ul>
  </body>
</html>
"""

cleaner = HTMLCleaner()
result = cleaner.clean_for_rag(html)

print("Cleaned text:")
print(result["text"])
print("Metadata:")
print(result["metadata"])
```

**Output:**
```
Cleaned text:
[H1] Welcome

This is a test document.

[UL START]
[ITEM 1] Item 1
[ITEM 2] Item 2
[UL END]
...

Metadata:
{'title': 'Sample HTML', 'author': 'Alice'}
```