"""
core.metadata.customer_adapter
============================

Adapter for customer-specific metadata configuration.

This module provides:
1. Loading customer metadata configuration from YAML
2. Creating customer-specific schemas and extractors
3. Adapting customer rules to metadata registry format
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from core.config.settings import settings
from core.services.customer_service import customer_service
from core.metadata.registry import (
    MetadataSchema,
    MetadataField,
    MetadataFieldType,
    register_schema,
    extraction_rule,
    extract_metadata
)

logger = logging.getLogger(__name__)

class CustomerMetadataAdapter:
    """Adapter for customer-specific metadata configuration."""

    def __init__(self, customer_id: str):
        """
        Initialize adapter for a customer.

        Args:
            customer_id: Customer identifier
        """
        self.customer_id = customer_id
        self.customer_config = customer_service.get_customer_config(customer_id) or {}
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load customer metadata configuration."""
        # Load customer config from their customer.yaml file
        customer_config = customer_service.get_customer_config(self.customer_id) or {}

        # Extract the metadata section
        metadata_config = customer_config.get("metadata", {})
        logger.info(f"Loaded metadata configuration for customer: {self.customer_id}")

        return metadata_config

    def register_schemas(self) -> List[str]:
        """
        Register customer-specific metadata schemas.

        Returns:
            List of registered schema names
        """
        registered_schemas = []

        # Extract any custom schemas defined in the customer config
        schemas_config = self.config.get("schemas", {})
        for schema_name, schema_def in schemas_config.items():
            # Create a customer-prefixed schema name to avoid collisions
            prefixed_name = f"{self.customer_id}_{schema_name}"

            # Create schema
            schema = MetadataSchema(
                name=prefixed_name,
                description=schema_def.get("description", f"Custom schema for {self.customer_id}")
            )

            # Add fields from config
            for field_name, field_def in schema_def.get("fields", {}).items():
                field_type_str = field_def.get("type", "string")
                try:
                    field_type = MetadataFieldType(field_type_str)
                except ValueError:
                    logger.warning(f"Invalid field type: {field_type_str}, using STRING")
                    field_type = MetadataFieldType.STRING

                # Create the field
                field = MetadataField(
                    name=field_name,
                    field_type=field_type,
                    description=field_def.get("description", ""),
                    required=field_def.get("required", False),
                    default=field_def.get("default"),
                    pattern=field_def.get("pattern")
                )

                schema.add_field(field)

            # Register the schema
            register_schema(schema)
            registered_schemas.append(prefixed_name)
            logger.info(f"Registered customer schema: {prefixed_name}")

        # Register dataset-specific schemas
        self._register_dataset_schemas()

        # Register purpose patterns from customer config
        self._register_purpose_patterns()

        return registered_schemas

    def _register_dataset_schemas(self) -> None:
        """Register metadata schemas from dataset configurations."""
        # Get datasets configuration from customer config
        datasets_config = self.customer_config.get("datasets", {})
        
        # Log for debugging
        logger.debug(f"Registering dataset schemas for customer {self.customer_id}")
        logger.debug(f"Found {len(datasets_config)} dataset configurations")
        
        for dataset_name, dataset_config in datasets_config.items():
            # Check if dataset has a metadata_schema definition
            metadata_schema = dataset_config.get("metadata_schema")
            if not metadata_schema:
                continue
                
            logger.info(f"Found metadata schema for dataset {dataset_name}")
            
            # Create a schema name with customer and dataset prefixes
            schema_name = f"{self.customer_id}_{dataset_name}"
            
            # Create the schema
            schema = MetadataSchema(
                name=schema_name,
                description=f"Schema for {self.customer_id} {dataset_name} dataset"
            )
            
            # Add fields from the schema definition
            for field_name, field_def in metadata_schema.items():
                field_type_str = field_def.get("type", "string").lower()
                try:
                    field_type = MetadataFieldType(field_type_str)
                except ValueError:
                    logger.warning(f"Invalid field type '{field_type_str}' for field '{field_name}', defaulting to string")
                    field_type = MetadataFieldType.STRING
                    
                field = MetadataField(
                    name=field_name,
                    field_type=field_type,
                    description=field_def.get("description", ""),
                    required=field_def.get("required", False),
                    default=field_def.get("default"),
                    pattern=field_def.get("pattern")
                )
                
                schema.add_field(field)
                
            # Register the schema
            register_schema(schema)
            logger.info(f"Registered dataset schema: {schema_name} with {len(metadata_schema)} fields")
            
            # Update settings to indicate the schema is registered
            # This is critical to make the schema available to loaders
            settings_datasets = settings.get("datasets", {})
            if isinstance(settings_datasets, dict) and dataset_name in settings_datasets:
                if not isinstance(settings_datasets[dataset_name], dict):
                    settings_datasets[dataset_name] = {}
                settings_datasets[dataset_name]["_metadata_schema_registered"] = True
                settings["datasets"] = settings_datasets
                logger.info(f"Updated settings for dataset {dataset_name} to mark schema as registered")
            
            # Store the schema in settings so it can be accessed by loaders
            datasets = settings.get("datasets", {})
            if dataset_name in datasets:
                datasets[dataset_name]["_metadata_schema_registered"] = True
                settings["datasets"] = datasets

    def _register_purpose_patterns(self) -> None:
        """Register customer-specific purpose patterns as extraction rules."""
        # Get purpose patterns from config
        purpose_patterns = self.config.get("extractor", {}).get("purpose_patterns", {})

        if not purpose_patterns:
            return

        # Dynamic function generation for customer-specific extraction rules
        for purpose, patterns in purpose_patterns.items():
            # Skip if not a list of patterns
            if not isinstance(patterns, list):
                continue

            # Create a content purpose extraction rule based on customer patterns
            @extraction_rule("content_purpose")
            def customer_purpose_rule(text: str, context: Dict[str, Any], _purpose=purpose, _patterns=patterns) -> Optional[str]:
                """Customer-specific content purpose extraction rule."""
                text_lower = text.lower()
                score = 0

                for pattern in _patterns:
                    matches = re.findall(pattern, text_lower)
                    score += len(matches)

                # Return the purpose if matches found
                if score > 0:
                    # Check if customer context matches
                    if context.get("customer_id") == self.customer_id:
                        return _purpose

                return None

    def extract_metadata(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract metadata using customer-specific schemas.

        Args:
            text: Document text
            context: Additional context

        Returns:
            Extracted metadata dict
        """
        if context is None:
            context = {}
            context = {}

        # Add customer ID to context
        context["customer_id"] = self.customer_id

        # Add customer ID to context
        context["customer_id"] = self.customer_id

        # Get customer-specific schemas
        customer_schemas = [f"{self.customer_id}_{schema}" for schema in self.config.get("schemas", {}).keys()]
        
        # Add dataset schemas
        dataset_schemas = []
        for dataset_name in self.customer_config.get("datasets", {}).keys():
            if self.customer_config.get("datasets", {}).get(dataset_name, {}).get("metadata_schema"):
                dataset_schemas.append(f"{self.customer_id}_{dataset_name}")
        
        # Include core, domain, customer and dataset schemas
        schemas = ["core", "content"] + customer_schemas + dataset_schemas

        # Extract metadata
        metadata = extract_metadata(text, context, schemas)

        # Apply any customer-specific post-processing
        metadata = self._apply_customer_processors(metadata)

        return metadata

    def _apply_customer_processors(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply customer-specific processing to extracted metadata."""
        # Apply customer-specific processors based on configuration
        custom_opts = self.config.get("extractor", {}).get("custom", {})

        # Example: add domain-specific keywords if configured
        if custom_opts.get("domain_specific_keywords", False):
            product_domains = settings.get("SUBJECT_DOMAINS", {})

            # Find matching domains based on metadata content
            if "title" in metadata and isinstance(metadata["title"], str):
                title = metadata["title"].lower()
                domain_matches = []

                for domain, config in product_domains.items():
                    keywords = config.get("keywords", [])
                    for keyword in keywords:
                        if isinstance(keyword, list) and len(keyword) > 0:
                            if keyword[0].lower() in title:
                                domain_matches.append(domain)
                                break
                        elif isinstance(keyword, str) and keyword.lower() in title:
                            domain_matches.append(domain)
                            break

                if domain_matches:
                    metadata["product_domains"] = domain_matches

        # Add any other customer-specific processing here
        return metadata
