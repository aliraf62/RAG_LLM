import pytest
from langchain.schema import Document
from core.pipeline.chunkers import HTMLChunker
from core.pipeline.chunkers import MarkdownChunker
from core.pipeline.chunkers import TextChunker
@pytest.mark.timeout(5)
@pytest.mark.parametrize("cls,input_text", [
    (HTMLChunker, "<h1>Hi</h1><p>There</p>"),
    (MarkdownChunker, "# Hi\n\nThere"),
    (TextChunker, "Line one.\nLine two."),
])
def test_chunker_returns_documents(cls, input_text):
    chunker = cls()
    # Add metadata parameter with required doc_id
    metadata = {"doc_id": "test_doc"}
    docs = chunker.chunk(content=input_text, metadata=metadata)
    assert isinstance(docs, list)
    assert all(isinstance(d, Document) for d in docs)
    # Every doc should carry a chunk_type in metadata
    assert all("chunk_type" in d.metadata for d in docs)
    # And none should be empty
    assert all(d.page_content.strip() for d in docs)