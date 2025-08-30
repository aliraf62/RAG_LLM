"""Basic functionality tests for data ingestion cleaners."""
import unittest
import logging

from core.pipeline import (
    clean_content,
    detect_content_type,
    HTMLCleaner,
    MarkdownCleaner,
    TextCleaner
)

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CleanersBasicTest(unittest.TestCase):
    """Basic functionality tests for different content cleaners."""

    def test_html_cleaner(self):
        """Test that HTML cleaner works correctly."""
        html = """
        <html>
            <head>
                <title>Test Document</title>
                <meta name="description" content="A test HTML document">
            </head>
            <body>
                <h1>Hello World</h1>
                <p>This is <b>bold</b> text.</p>
                <div>
                    <ul>
                        <li>Item 1</li>
                        <li>Item 2</li>
                    </ul>
                </div>
                <script>alert('hidden');</script>
            </body>
        </html>
        """
        cleaner = HTMLCleaner()
        result = cleaner.clean_for_rag(html)

        # Verify basic cleaning worked
        self.assertIn("Hello World", result["text"])
        self.assertIn("This is bold text", result["text"])
        self.assertIn("Item 1", result["text"])

        # Verify script was removed
        self.assertNotIn("alert", result["text"])

        # Check metadata extraction
        self.assertIn("metadata", result)
        self.assertEqual("Test Document", result["metadata"].get("title"))

        logger.info(f"HTML cleaner produced text: {result['text'][:100]}...")

    def test_markdown_frontmatter_variants(self):
        """Test different frontmatter formats in markdown."""
        # Test tabbed frontmatter
        tabbed_markdown = """---
        title: Tabbed Format
        author: Test Author
        ---
        # Content"""

        # Test non-tabbed frontmatter
        non_tabbed_markdown = """---
    title: Non-tabbed Format
    author: Test Author
    ---
    # Content"""

        cleaner = MarkdownCleaner()
        tabbed_result = cleaner.clean_for_rag(tabbed_markdown)
        non_tabbed_result = cleaner.clean_for_rag(non_tabbed_markdown)

        self.assertIn("frontmatter", tabbed_result)
        self.assertIn("frontmatter", non_tabbed_result)
        self.assertEqual("Tabbed Format", tabbed_result["frontmatter"].get("title"))
        self.assertEqual("Non-tabbed Format", non_tabbed_result["frontmatter"].get("title"))

    def test_markdown_cleaner(self):
        """Test that Markdown cleaner works correctly."""
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

        # Verify basic cleaning worked
        self.assertIn("Heading 1", result["text"])
        self.assertIn("bold text", result["text"])
        self.assertIn("List item", result["text"])

        # Verify frontmatter extraction
        self.assertIn("frontmatter", result)
        self.assertEqual("Test Markdown", result["frontmatter"].get("title"))
        self.assertEqual("Test Author", result["frontmatter"].get("author"))

        logger.info(f"Markdown cleaner produced text: {result['text'][:100]}...")

    def test_text_cleaner(self):
        """Test that Text cleaner works correctly."""
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

        # Verify basic cleaning worked
        self.assertIn("Sample Document", result["text"])
        self.assertIn("multiple paragraphs", result["text"])

        # Check metadata extraction
        self.assertIn("metadata", result)
        self.assertEqual("Sample Document", result["metadata"].get("title"))
        self.assertEqual("Test User", result["metadata"].get("author"))

        # Verify sentence segmentation
        self.assertIn("sentences", result)
        self.assertGreater(len(result["sentences"]), 2)

        logger.info(f"Text cleaner produced text: {result['text'][:100]}...")

    def test_content_detection(self):
        """Test that content type detection works correctly."""
        html = "<html><body><h1>Test</h1><p>Content</p></body></html>"
        markdown = "# Heading\n\nParagraph with **bold** and _italic_ text."
        text = "Just some plain text content without any markup."

        self.assertEqual("html", detect_content_type(html))
        self.assertEqual("markdown", detect_content_type(markdown))
        self.assertEqual("text", detect_content_type(text))

    def test_combined_clean_content(self):
        """Test the combined clean_content function with auto-detection."""
        html = "<html><body><h1>Test</h1><p>Content</p></body></html>"
        result = clean_content(html)

        self.assertIn("Test", result["text"])
        self.assertIn("Content", result["text"])
        self.assertEqual("text", result["format"])  # Default format is text

        # Try markdown
        markdown = "# Test Heading\n\nThis is a paragraph."
        result = clean_content(markdown)
        self.assertIn("Test Heading", result["text"])


if __name__ == "__main__":
    unittest.main()