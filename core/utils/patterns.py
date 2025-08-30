"""
Centralized regex patterns for text, markdown, and HTML processing.

Defines and documents all regular expressions used throughout the project for
content detection, formatting, extraction, cleaning, tokenization, and normalization.
See docs/architecture/patterns.md for design rationale and pattern catalog.
"""
import re

# ---------------------------------------------------------------------
# Content Detection Patterns
# ---------------------------------------------------------------------

# Matches HTML tags like <div>, <p>, <h1>, etc.
# Example: "<div class='content'>", "<p>Text</p>"
HTML_PATTERN = re.compile(r'</?(?:div|span|p|h[1-6]|ul|ol|li|table|html|body|head)(?:\s|>)')

# Matches Markdown headings (# Heading)
# Example: "# Title", "### Subsection"
MARKDOWN_HEADING_PATTERN = re.compile(r'^#{1,6}\s+\w+', re.MULTILINE)

# Matches Markdown list items (* Item or - Item)
# Example: "* List item", "- Another item"
MARKDOWN_LIST_PATTERN = re.compile(r'^\s*[-*+]\s+\w+', re.MULTILINE)

# Matches Markdown code block markers
# Example: "```python", "```"
MARKDOWN_CODE_PATTERN = re.compile(r'^```\w*$', re.MULTILINE)


# ---------------------------------------------------------------------
# Markdown Formatting Patterns
# ---------------------------------------------------------------------

# Matches Markdown headings to remove them
# Example: "# Heading" -> ""
MARKDOWN_HEADING_REMOVE_PATTERN = re.compile(r'^#{1,6}\s+', re.MULTILINE)

# Matches Markdown emphasis formatting for removal
# Example: "*emphasized*" -> "emphasized"
MARKDOWN_EMPHASIS_PATTERN = re.compile(r'[*_]{1,2}([^*_]+)[*_]{1,2}')

# Matches Markdown inline code formatting for removal
# Example: "`code`" -> "code"
MARKDOWN_INLINE_CODE_PATTERN = re.compile(r'`([^`]+)`')

# Extracts the first heading from Markdown content (for title extraction)
# Example: "# Document Title" -> "Document Title"
MARKDOWN_FIRST_HEADING_PATTERN = re.compile(r'^#\s+(.+)$', re.MULTILINE)

# ---------------------------------------------------------------------
# Extraction Patterns
# ---------------------------------------------------------------------

# Matches YAML frontmatter in Markdown documents, handling indented content
# Example: "---\ntitle: Document\nauthor: Name\n---"
MARKDOWN_FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n\s*---\s*\n', re.DOTALL)

# Extracts metadata from HTML meta tags
# Example: <meta name="author" content="John Doe">
HTML_META_TAG_PATTERN = re.compile(r'<meta\s+name="([^"]+)"\s+content="([^"]*)"', re.I)

# Extracts image src from HTML img tags
# Example: <img src="image.png" alt="desc">
HTML_IMG_SRC_PATTERN = re.compile(r'<img\s+[^>]*src=["\']([^"\']+)["\']', re.IGNORECASE)

# Extracts image references from Markdown image syntax
# Example: ![alt text](image.png)
MARKDOWN_IMAGE_PATTERN = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')

# Extracts the title from HTML <title> tags
# Example: <title>Document Title</title>
HTML_TITLE_PATTERN = re.compile(r'<title>([^<]+)</title>', re.IGNORECASE)

# Extracts headings from HTML (level and text)
# Example: <h2>Section</h2>
HTML_HEADING_PATTERN = re.compile(r'<h([1-6])[^>]*>([^<]+)</h\1>', re.IGNORECASE)

# Extracts headings from Markdown for outline (level and text)
# Example: "## Section Title"
MARKDOWN_HEADING_OUTLINE_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# Extracts code block languages from Markdown
# Example: "```python"
MARKDOWN_CODE_LANGUAGE_PATTERN = re.compile(r'```(\w+)')

# Detects PDF page markers in text
# Example: "Page 2 of 10"
PDF_PAGE_MARKER_PATTERN = re.compile(r'\bpage\s+\d+\s+of\s+\d+\b', re.IGNORECASE)

# ---------------------------------------------------------------------
# Cleaning Patterns
# ---------------------------------------------------------------------

# Matches Markdown heading patterns for processing
# Example: "## Heading 2" -> capture groups: ("##", "Heading 2")
MARKDOWN_HEADING_FORMAT_PATTERN = re.compile(r'^(#+)\s+(.+)$', re.MULTILINE)

# Matches Markdown list items for processing
# Example: "* Item" or "1. Item" -> capture groups: ("*" or "1.", "Item")
MARKDOWN_LIST_ITEM_PATTERN = re.compile(r'^([\*\-+]|\d+\.)\s+(.+)$', re.MULTILINE)

# Matches Markdown code blocks with language for processing
# Example: "```python\ndef func():\n    pass\n```"
MARKDOWN_CODE_BLOCK_PATTERN = re.compile(r'```(?:\w+)?\n(.*?)```', re.DOTALL)

# Matches text that looks like headings for text chunking
# Example: "# Heading", "** Bold heading", "1. Numbered heading"
TEXT_HEADING_PATTERN = re.compile(r"^(#+|\*\*?|\d+\.)\s+", re.M)
# Example: "Title: Document Title"
TEXT_TITLE_PATTERN = re.compile(r"(?:Title|TITLE):\s*(.+)(?:\n|$)")
TEXT_AUTHOR_PATTERN = re.compile(r"(?:Author|AUTHOR):\s*(.+)(?:\n|$)")
TEXT_DATE_PATTERN = re.compile(r"(?:Date|DATE):\s*(.+)(?:\n|$)")
TEXT_SUBJECT_PATTERN = re.compile(r"(?:Subject|SUBJECT):\s*(.+)(?:\n|$)")

# Matches inline links in Markdown
# Example: "[text](https://example.com)"
MARKDOWN_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

# Matches HTML anchor tags to convert to text
# Example: <a href="https://example.com">Link text</a>
HTML_ANCHOR_TAG_PATTERN = re.compile(r'<a\s+[^>]*href=["\\\']([^"\\\']+)["\\\'][^>]*>(.*?)</a>', re.IGNORECASE)

# Pattern for sanitizing filenames (removes characters unsafe for filesystems)
FILENAME_SANITIZE_PATTERN = re.compile(r'[^\w\s.-]')

# Matches common video hosting platforms
# Example: "https://vimeo.com/123456"
VIDEO_URL_PATTERN = re.compile(
    r'https?://(?:'
    r'(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)|'
    r'(?:www\.)?vimeo\.com/|'
    r'(?:www\.)?loom\.com/share/|'
    r'(?:www\.)?wistia\.com/|'
    r'(?:player\.)?vimeo\.com/video/|'
    r'(?:www\.)?dailymotion\.com/video/)'
    r'[a-zA-Z0-9_\-]+',
    re.IGNORECASE
)

# ---------------------------------------------------------------------
# Tokenization Patterns
# ---------------------------------------------------------------------

# Tokenizes text into words and punctuation (for chunking)
# Example: "Hello, world!" -> ["Hello", ",", "world", "!"]
WORD_TOKENIZATION_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Splits text into sentences
# Example: "Hello. How are you?" -> ["Hello", "How are you?"]
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')

# ---------------------------------------------------------------------
# Text Normalization Patterns
# ---------------------------------------------------------------------

# Normalizes whitespace (multiple spaces to single space)
# Example: "text    with    spaces" -> "text with spaces"
WHITESPACE_NORMALIZATION_PATTERN = re.compile(r'\s+')

# Normalizes newlines (multiple newlines to single newline)
# Example: "text\n\n\nwith\nlines" -> "text\nwith\nlines" 
NEWLINE_NORMALIZATION_PATTERN = re.compile(r'(\n\s*)+')

# Keyword matching pattern (used in classifications)
# Example: finding "keyword" in "text with keyword inside"
def keyword_pattern(keyword):
    """Create a regex pattern that matches a keyword as a complete word.

    Args:
        keyword (str): The keyword to match.

    Returns:
        Pattern: Compiled regex pattern for the keyword.
    """
    return re.compile(r'\b' + re.escape(keyword) + r'\b')