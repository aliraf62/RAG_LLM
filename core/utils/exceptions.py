"""
core.utils.exceptions
====================

Custom exceptions for the RAG pipeline and application components.
"""

class CoreException(Exception):
    """Base exception for all core module errors."""
    pass

class ConfigurationError(CoreException):
    """Error in configuration settings."""
    pass

class EmbeddingError(CoreException):
    """Error during embedding generation."""
    pass

class RetrievalError(CoreException):
    """Error during document retrieval."""
    pass

class LLMError(CoreException):
    """Error during LLM interaction."""
    pass

class IndexError(CoreException):
    """Error related to vector index operations."""
    pass

class VectorStoreError(CoreException):
    """Error related to vector store retriever operations."""
    pass

class ExporterError(CoreException):
    """Error during document exporting."""
    pass

class ExtractorError(CoreException):
    """Error during document extraction."""
    pass

class LoaderError(CoreException):
    """Error during document loading."""
    pass

class ChunkerError(CoreException):
    """Error during document chunking."""
    pass

class RAGError(CoreException):
    """Error during RAG processing."""
    pass

class ClassificationError(CoreException):
    """Error during query classification."""
    pass

class AssetProcessingError(CoreException):
    """Error during asset processing"""
    pass

# Authentication-related exceptions
class AuthenticationError(CoreException):
    """Exception raised for authentication failures."""
    pass

class AuthorizationError(CoreException):
    """Exception raised when a user lacks permission for an operation."""
    pass

class SessionError(CoreException):
    """Exception raised for session-related errors."""
    pass

