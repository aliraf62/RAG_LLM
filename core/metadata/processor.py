"""
core.metadata.processor
=====================

Process and integrate metadata throughout the RAG pipeline.

This module:
1. Serves as the entry point for metadata processing in the ingestion pipeline
2. Applies metadata enrichment during retrieval
3. Ensures metadata is properly structured for LLM context
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.config.settings import settings
from core.services.customer_service import customer_service
from core.metadata.registry import extract_metadata, MetadataRegistry
from core.metadata.customer_adapter import CustomerMetadataAdapter
from core.utils.component_registry import register
from core.config.paths import find_project_root

logger = logging.getLogger(__name__)

# Initialize metadata registry with core schemas
_SCHEMA_DIR = find_project_root() / "core" / "metadata" / "schemas"
metadata_registry = MetadataRegistry()

# Register core schemas
metadata_registry.load_schema("content", _SCHEMA_DIR / "content.yaml")

# Register canonical keys from settings
if hasattr(settings, 'CANONICAL_METADATA_KEYS'):
    logger.info(f"Registering canonical keys from settings: {settings.CANONICAL_METADATA_KEYS}")
    for key in settings.CANONICAL_METADATA_KEYS:
        if not metadata_registry.has_field(key):
            metadata_registry.register_field(key, "string", f"Canonical field: {key}")


@register("metadata_processor", "default")
class MetadataProcessor:
    """Process document metadata throughout the RAG pipeline."""

    def __init__(self, customer_id: Optional[str] = None):
        """
        Initialize the metadata processor.

        Args:
            customer_id: Optional customer identifier
        """
        self.customer_id = customer_id
        self.customer_adapter = None

        # Initialize customer adapter if specified
        if customer_id:
            try:
                self.customer_adapter = CustomerMetadataAdapter(customer_id)
                # Register customer-specific schemas
                self.customer_adapter.register_schemas()
            except Exception as e:
                logger.warning(f"Failed to initialize customer metadata adapter: {e}")

    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document during ingestion pipeline.

        Args:
            document: Document dictionary containing text and any existing metadata

        Returns:
            Document with enriched metadata
        """
        # Extract document content and existing metadata
        text = document.get("text", document.get("page_content", ""))
        metadata = document.get("metadata", {}).copy()

        # Gather context for extraction
        context = self._build_extraction_context(document)

        # Extract new metadata from content
        if self.customer_adapter:
            # Use customer-specific extraction
            new_metadata = self.customer_adapter.extract_metadata(text, context)
        else:
            # Use default extraction (core schemas only)
            new_metadata = extract_metadata(text, context, ["core", "content"])

        # Merge existing metadata with extracted metadata (existing takes precedence)
        for key, value in new_metadata.items():
            if key not in metadata or metadata[key] is None:
                metadata[key] = value

        # Ensure canonical keys are present (may be None if not applicable)
        if hasattr(settings, 'CANONICAL_METADATA_KEYS'):
            for key in settings.CANONICAL_METADATA_KEYS:
                if key not in metadata:
                    metadata[key] = None

        # Update document with enriched metadata
        document["metadata"] = metadata
        return document

    def _build_extraction_context(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Build context dictionary for metadata extraction."""
        context = {}

        # Add filename if available
        if "source" in document:
            source = document["source"]
            if isinstance(source, (str, Path)):
                source_path = Path(source)
                context["filename"] = source_path.name
                context["file_extension"] = source_path.suffix.lstrip(".")

        # Add customer ID if available
        if self.customer_id:
            context["customer_id"] = self.customer_id

        # Add any other relevant context from document
        if "dataset_id" in document:
            context["dataset_id"] = document["dataset_id"]

        return context

    def enrich_retrieval_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich metadata in retrieval results.

        This can be called after document retrieval to enhance metadata before
        passing to the LLM for response generation.

        Args:
            results: List of retrieval results

        Returns:
            Enriched retrieval results
        """
        enriched_results = []

        for result in results:
            # Process each retrieval result
            metadata = result.get("metadata", {}).copy()

            # Add relevance indicators if available
            if "score" in result:
                score = result["score"]
                metadata["relevance_score"] = score

                # Add qualitative assessment based on threshold
                threshold = settings.get("SIMILARITY_THRESHOLD", 0.2)
                if score > threshold * 2:
                    metadata["relevance"] = "high"
                elif score > threshold:
                    metadata["relevance"] = "medium"
                else:
                    metadata["relevance"] = "low"

            # Apply document type formatting based on metadata
            if "document_type" in metadata:
                doc_type = metadata["document_type"]
                if doc_type == "guide":
                    metadata["type_indicator"] = "📚 Guide"
                elif doc_type == "tutorial":
                    metadata["type_indicator"] = "🔍 Tutorial"
                elif doc_type == "reference":
                    metadata["type_indicator"] = "📖 Reference"
                elif doc_type == "faq":
                    metadata["type_indicator"] = "❓ FAQ"

            # Add content purpose if available
            if "content_purpose" in metadata:
                purpose = metadata["content_purpose"]
                if purpose == "how_to":
                    metadata["purpose_indicator"] = "Step-by-step guide"
                elif purpose == "explanation":
                    metadata["purpose_indicator"] = "Concept explanation"
                elif purpose == "reference":
                    metadata["purpose_indicator"] = "Reference material"

            # Update result with enriched metadata
            result["metadata"] = metadata
            enriched_results.append(result)

        return enriched_results

    def format_metadata_for_llm(self, metadata: Dict[str, Any], include_fields: List[str] = None) -> str:
        """
        Format document metadata for inclusion in LLM context.

        Args:
            metadata: Document metadata
            include_fields: List of fields to include (None for all)

        Returns:
            Formatted metadata string for LLM context
        """
        # Filter fields if specified
        if include_fields:
            metadata = {k: v for k, v in metadata.items() if k in include_fields}

        formatted_parts = []

        # Add title if available
        if "title" in metadata or "html_title" in metadata:
            title = metadata.get("title") or metadata.get("html_title", "")
            formatted_parts.append(f"Title: {title}")

        # Add document type indicators
        if "type_indicator" in metadata:
            formatted_parts.append(metadata["type_indicator"])

        # Add purpose indicator
        if "purpose_indicator" in metadata:
            formatted_parts.append(metadata["purpose_indicator"])

        # Add source if available
        if "source" in metadata:
            source = metadata["source"]
            formatted_parts.append(f"Source: {source}")

        # Add author if available
        if "authors" in metadata and metadata["authors"]:
            if isinstance(metadata["authors"], list):
                authors = ", ".join(metadata["authors"])
            else:
                authors = metadata["authors"]
            formatted_parts.append(f"Author: {authors}")

        # Add date if available
        for date_field in ["created_at", "updated_at"]:
            if date_field in metadata:
                formatted_date = metadata[date_field]
                field_name = "Created" if date_field == "created_at" else "Updated"
                formatted_parts.append(f"{field_name}: {formatted_date}")
                break

        return "\n".join(formatted_parts)

    def apply_metadata_to_context(self, context_text: str, metadata: Dict[str, Any]) -> str:
        """
        Apply metadata to the retrieval context for the LLM.

        Args:
            context_text: Original retrieval context text
            metadata: Document metadata

        Returns:
            Enhanced context with metadata information
        """
        formatted_metadata = self.format_metadata_for_llm(metadata)

        # Don't add empty metadata
        if not formatted_metadata:
            return context_text

        # Add metadata section at the beginning of the context
        return f"{formatted_metadata}\n\n{context_text}"


# Convenience functions
def process_document_metadata(document: Dict[str, Any], customer_id: Optional[str] = None) -> Dict[str, Any]:
    """Process metadata for a single document."""
    processor = MetadataProcessor(customer_id)
    return processor.process_document(document)

def process_batch_metadata(documents: List[Dict[str, Any]], customer_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Process metadata for a batch of documents."""
    processor = MetadataProcessor(customer_id)
    return [processor.process_document(doc) for doc in documents]
