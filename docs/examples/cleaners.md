# Example: Using Data Cleaners

This example demonstrates how to use the data cleaners (`HTMLCleaner`, `MarkdownCleaner`, `TextCleaner`) and the unified `clean_content` function to process raw content for RAG pipelines.

---

## 1. Unified Cleaning with `clean_content`

```python
from core.pipeline import clean_content

# Example HTML
html = """
<html>
  <head>
    <title>Test Document</title>
    <meta name="description" content="A test HTML document">
  </head>
  <body>
    <h1>Hello World</h1>
    <p>This is <b>bold</b> text.</p>
    <ul>
      <li>Item 1</li>
      <li>Item 2</li>
    </ul>
    <script>alert('hidden');</script>
  </body>
</html>
"""

result = clean_content(html)
print("Cleaned text:", result["text"])
print("Metadata:", result["metadata"])
```

---

## 2. Cleaning Markdown

```python
from core.pipeline import MarkdownCleaner

markdown = """---
title: Test Markdown
author: Test Author
---

# Heading 1

This is **bold** text with a [link](https://example.com).

* List item 1
* List item 2

```python
def hello():
    print("Hello World")
```
"""

cleaner = MarkdownCleaner()
result = cleaner.clean_for_rag(markdown)
print("Cleaned text:", result["text"])
print("Frontmatter:", result.get("frontmatter"))
```

---

## 3. Cleaning Plain Text

```python
from data_ingestion.cleaners import TextCleaner

text = """
Title: Sample Document
Author: Test User

This is a sample text document.
It has multiple paragraphs.

And some extra spacing.

The end.
"""

cleaner = TextCleaner()
result = cleaner.clean_for_rag(text)
print("Cleaned text:", result["text"])
print("Metadata:", result["metadata"])
print("Sentences:", result.get("sentences"))
```

---

## 4. Auto-detection

You can use `clean_content` without specifying the content type; it will auto-detect HTML, Markdown, or plain text.

```python
from core.pipeline import clean_content

markdown = "# Heading\n\nParagraph with **bold** and _italic_ text."
result = clean_content(markdown)
print("Detected format:", result["format"])
print("Cleaned text:", result["text"])
```

---

## See Also

- [Guide: Data Cleaners](../guides/data_cleaners.md)
