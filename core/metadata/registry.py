"""
core.metadata.registry
=====================

Registry for metadata schemas, fields, and extraction rules.

This module provides a central registry for defining, validating, and
extracting document metadata throughout the RAG pipeline.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import yaml
from pathlib import Path
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Type aliases for better readability
MetadataValue = Union[str, int, float, bool, List[str], Dict[str, Any]]
MetadataDict = Dict[str, MetadataValue]
ExtractionRule = Callable[[str, Dict[str, Any]], Optional[MetadataValue]]


class MetadataFieldType(Enum):
    """Defines the data types supported for metadata fields."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    STRING_LIST = "string_list"
    OBJECT = "object"  # Nested metadata


class MetadataField:
    """Definition of a metadata field with validation rules."""

    def __init__(
        self,
        name: str,
        field_type: MetadataFieldType,
        description: str = "",
        required: bool = False,
        default: Any = None,
        pattern: Optional[str] = None,
        extraction_rules: Optional[List[ExtractionRule]] = None,
        nested_fields: Optional[Dict[str, 'MetadataField']] = None
    ):
        """
        Initialize a metadata field definition.

        Args:
            name: Field identifier (e.g. 'title')
            field_type: Data type for the field
            description: Human-readable description
            required: Whether the field is required
            default: Default value if not provided
            pattern: Regex pattern for validation (for string types)
            extraction_rules: Functions to extract this field from content
            nested_fields: For object types, defines nested fields
        """
        self.name = name
        self.field_type = field_type
        self.description = description
        self.required = required
        self.default = default
        self.pattern = pattern
        self.extraction_rules = extraction_rules or []
        self.nested_fields = nested_fields or {}

        # Validate default value matches type
        if default is not None:
            self.validate(default)

    def validate(self, value: Any) -> bool:
        """
        Validate a value against this field's rules.

        Args:
            value: Value to validate

        Returns:
            True if valid, False otherwise

        Raises:
            TypeError: If value doesn't match field_type
            ValueError: If value doesn't match pattern
        """
        if value is None:
            return not self.required

        # Type validation
        if self.field_type == MetadataFieldType.STRING:
            if not isinstance(value, str):
                raise TypeError(f"Expected string for {self.name}, got {type(value)}")
            if self.pattern and not re.match(self.pattern, value):
                raise ValueError(f"Value '{value}' doesn't match pattern {self.pattern}")

        elif self.field_type == MetadataFieldType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"Expected integer for {self.name}, got {type(value)}")

        elif self.field_type == MetadataFieldType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"Expected float for {self.name}, got {type(value)}")

        elif self.field_type == MetadataFieldType.BOOLEAN:
            if not isinstance(value, bool):
                raise TypeError(f"Expected boolean for {self.name}, got {type(value)}")

        elif self.field_type == MetadataFieldType.DATETIME:
            if not isinstance(value, (dt.datetime, dt.date, str)):
                raise TypeError(f"Expected datetime or string for {self.name}")
            if isinstance(value, str):
                try:
                    dt.datetime.fromisoformat(value)
                except ValueError:
                    raise ValueError(f"Expected ISO format datetime string for {self.name}")

        elif self.field_type == MetadataFieldType.STRING_LIST:
            if not isinstance(value, list):
                raise TypeError(f"Expected list for {self.name}")
            for item in value:
                if not isinstance(item, str):
                    raise TypeError(f"Expected string items in list for {self.name}")

        elif self.field_type == MetadataFieldType.OBJECT:
            if not isinstance(value, dict):
                raise TypeError(f"Expected dict for {self.name}")
            # Validate nested fields
            for nested_name, nested_field in self.nested_fields.items():
                if nested_name in value:
                    nested_field.validate(value[nested_name])

        return True

    def extract(self, text: str, context: Dict[str, Any]) -> Optional[MetadataValue]:
        """
        Extract field value from text using extraction rules.

        Args:
            text: Content to extract from
            context: Additional context for extraction (e.g. filename)

        Returns:
            Extracted value or None if extraction failed
        """
        for rule in self.extraction_rules:
            try:
                value = rule(text, context)
                if value is not None:
                    return value
            except Exception as e:
                logger.warning(f"Error in extraction rule for {self.name}: {e}")

        return None

    def to_dict(self) -> Dict[str, Union[str, bool, Dict[str, Any]]]:
        """Convert field definition to dictionary for serialization.

        Returns:
            Dictionary with properly serialized values
        """
        result: Dict[str, Union[str, bool, Dict[str, Any]]] = {
            "name": self.name,
            "type": self.field_type.value,
            "description": self.description,
            "required": self.required
        }

        if self.default is not None:
            result["default"] = str(self.default) if not isinstance(self.default, (bool, int, float)) else self.default

        if self.pattern:
            result["pattern"] = self.pattern

        if self.nested_fields:
            # Convert nested fields to string-serialized format
            result["nested_fields"] = {
                str(name): str(field.to_dict()) for name, field in self.nested_fields.items()
            }

        return result


class MetadataSchema:
    """Defines a schema for document metadata."""

    def __init__(self, name: str, description: str = ""):
        """
        Initialize a metadata schema.

        Args:
            name: Schema identifier
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self.fields: Dict[str, MetadataField] = {}

    def add_field(self, field: MetadataField) -> None:
        """
        Add a field to the schema.

        Args:
            field: Field definition to add
        """
        self.fields[field.name] = field

    def validate(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate metadata against this schema.

        Args:
            metadata: Metadata dictionary to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If required fields are missing
        """
        # Check required fields
        for name, field in self.fields.items():
            if field.required and name not in metadata:
                raise ValueError(f"Required field '{name}' is missing")

        # Validate provided fields
        for name, value in metadata.items():
            if name in self.fields:
                self.fields[name].validate(value)

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "fields": {name: field.to_dict() for name, field in self.fields.items()}
        }


class MetadataExtractor:
    """Extracts metadata from document content."""

    def __init__(self, schemas: Optional[List[MetadataSchema]] = None):
        """
        Initialize a metadata extractor.

        Args:
            schemas: List of schemas to use for extraction
        """
        self.schemas = schemas or []

    def extract(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract metadata from text.

        Args:
            text: Document text
            context: Additional context (e.g. filename, source)

        Returns:
            Dictionary of extracted metadata
        """
        context = context or {}
        metadata = {}

        # Extract metadata for each field in each schema
        for schema in self.schemas:
            for name, field in schema.fields.items():
                if name not in metadata:  # Don't override already extracted fields
                    value = field.extract(text, context)
                    if value is not None:
                        metadata[name] = value

        # Apply defaults for missing fields
        for schema in self.schemas:
            for name, field in schema.fields.items():
                if name not in metadata and field.default is not None:
                    metadata[name] = field.default

        return metadata

    def normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize metadata to standard format.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Normalized metadata dictionary
        """
        normalized = {}

        for name, value in metadata.items():
            # Skip None values
            if value is None:
                continue

            # Convert to appropriate types
            if isinstance(value, (list, tuple)):
                normalized[name] = [str(x) for x in value]
            elif isinstance(value, (dt.datetime, dt.date)):
                normalized[name] = value.isoformat()
            elif isinstance(value, (int, float, bool)):
                normalized[name] = value
            elif isinstance(value, dict):
                # Recursively normalize nested dictionaries
                normalized[name] = self.normalize_metadata(value)
            else:
                normalized[name] = str(value)

        return normalized


# Registry for metadata schemas
_SCHEMAS: Dict[str, MetadataSchema] = {}

# Registry for extraction rules
_EXTRACTION_RULES: Dict[str, List[ExtractionRule]] = {}

# Core/default metadata schema
_CORE_SCHEMA = MetadataSchema(
    name="core",
    description="Core metadata fields common to all documents"
)

# Add core fields to the schema
_CORE_SCHEMA.add_field(MetadataField(
    name="title",
    field_type=MetadataFieldType.STRING,
    description="Document title",
    required=False
))

_CORE_SCHEMA.add_field(MetadataField(
    name="source",
    field_type=MetadataFieldType.STRING,
    description="Source of the document (e.g. URL, file path)",
    required=False
))

_CORE_SCHEMA.add_field(MetadataField(
    name="created_at",
    field_type=MetadataFieldType.DATETIME,
    description="Document creation timestamp",
    required=False
))

_CORE_SCHEMA.add_field(MetadataField(
    name="updated_at",
    field_type=MetadataFieldType.DATETIME,
    description="Document last update timestamp",
    required=False
))

_CORE_SCHEMA.add_field(MetadataField(
    name="authors",
    field_type=MetadataFieldType.STRING_LIST,
    description="Document authors",
    required=False
))

_CORE_SCHEMA.add_field(MetadataField(
    name="tags",
    field_type=MetadataFieldType.STRING_LIST,
    description="Content tags",
    required=False
))

_CORE_SCHEMA.add_field(MetadataField(
    name="content_type",
    field_type=MetadataFieldType.STRING,
    description="Content type (e.g. text, html, markdown)",
    required=False
))

_CORE_SCHEMA.add_field(MetadataField(
    name="language",
    field_type=MetadataFieldType.STRING,
    description="Document language code (e.g. en, fr)",
    required=False,
    pattern=r"^[a-z]{2}(-[A-Z]{2})?$"
))

# Register the core schema
_SCHEMAS["core"] = _CORE_SCHEMA


def register_schema(schema: MetadataSchema) -> None:
    """
    Register a metadata schema.

    Args:
        schema: Schema to register
    """
    _SCHEMAS[schema.name] = schema
    logger.info(f"Registered metadata schema: {schema.name}")


def get_schema(name: str) -> MetadataSchema:
    """
    Get a registered schema by name.

    Args:
        name: Schema name

    Returns:
        Metadata schema

    Raises:
        KeyError: If schema doesn't exist
    """
    if name not in _SCHEMAS:
        raise KeyError(f"Metadata schema not found: {name}")
    return _SCHEMAS[name]


def register_extraction_rule(field_name: str, rule: ExtractionRule) -> None:
    """
    Register an extraction rule for a field.

    Args:
        field_name: Name of the field to extract
        rule: Function to extract the field value
    """
    if field_name not in _EXTRACTION_RULES:
        _EXTRACTION_RULES[field_name] = []
    _EXTRACTION_RULES[field_name].append(rule)
    logger.info(f"Registered extraction rule for field: {field_name}")


def get_extraction_rules(field_name: str) -> List[ExtractionRule]:
    """
    Get all extraction rules for a field.

    Args:
        field_name: Field name

    Returns:
        List of extraction rules
    """
    return _EXTRACTION_RULES.get(field_name, [])


def create_extractor(schema_names: List[str] = None) -> MetadataExtractor:
    """
    Create a metadata extractor for specified schemas.

    Args:
        schema_names: List of schema names to include

    Returns:
        Configured metadata extractor
    """
    if schema_names is None:
        schema_names = ["core"]

    schemas = []
    for name in schema_names:
        try:
            schemas.append(get_schema(name))
        except KeyError:
            logger.warning(f"Schema not found: {name}")

    return MetadataExtractor(schemas)


def extract_metadata(text: str, context: Dict[str, Any] = None, schema_names: List[str] = None) -> Dict[str, Any]:
    """
    Extract metadata from document text.

    Args:
        text: Document text
        context: Additional context
        schema_names: List of schema names to use

    Returns:
        Extracted metadata
    """
    extractor = create_extractor(schema_names)
    metadata = extractor.extract(text, context or {})
    return extractor.normalize_metadata(metadata)


# Register extraction rule decorator
def extraction_rule(field_name: str):
    """
    Decorator for registering metadata extraction rules.

    Args:
        field_name: Name of the field to extract

    Returns:
        Decorator function
    """
    def decorator(func: ExtractionRule) -> ExtractionRule:
        register_extraction_rule(field_name, func)
        return func
    return decorator


class MetadataRegistry:
    """Registry for metadata schemas and extraction rules (facade for module-level registries)."""

    def __init__(self):
        """Initialize registry with module-level registries."""
        self._schemas = _SCHEMAS
        self._extraction_rules = _EXTRACTION_RULES

    def has_field(self, field_name: str) -> bool:
        """
        Check if a field is registered in any schema.

        Args:
            field_name: Name of the field to check

        Returns:
            True if the field exists in any schema, False otherwise
        """
        for schema in self._schemas.values():
            if field_name in schema.fields:
                return True
        return False

    def register_field(self, field_name: str, field_type_str: str, description: str = "", required: bool = False):
        """
        Register a new field in the core schema.

        Args:
            field_name: Name of the field to register
            field_type_str: Type of the field as a string ('string', 'integer', etc.)
            description: Human-readable description of the field
            required: Whether the field is required
        """
        try:
            field_type = MetadataFieldType(field_type_str)
        except ValueError:
            logger.warning(f"Invalid field type '{field_type_str}' for field '{field_name}', defaulting to string")
            field_type = MetadataFieldType.STRING

        field = MetadataField(
            name=field_name,
            field_type=field_type,
            description=description,
            required=required
        )

        # Add to core schema
        if "core" in self._schemas:
            self._schemas["core"].add_field(field)
            logger.info(f"Registered field '{field_name}' in core schema")
        else:
            logger.warning(f"Core schema not found, can't register field '{field_name}'")
            # Create core schema if it doesn't exist
            core_schema = MetadataSchema(name="core", description="Core metadata fields")
            core_schema.add_field(field)
            self._schemas["core"] = core_schema
            logger.info(f"Created core schema and registered field '{field_name}'")

    @staticmethod
    def register_schema(schema: MetadataSchema) -> None:
        """Register a metadata schema."""
        register_schema(schema)

    @staticmethod
    def get_schema(name: str) -> MetadataSchema:
        """Get a registered schema by name."""
        return get_schema(name)

    @staticmethod
    def register_extraction_rule(field_name: str, rule: ExtractionRule) -> None:
        """Register an extraction rule for a field."""
        register_extraction_rule(field_name, rule)

    @staticmethod
    def get_extraction_rules(field_name: str) -> List[ExtractionRule]:
        """Get all extraction rules for a field."""
        return get_extraction_rules(field_name)

    @staticmethod
    def extract_metadata(text: str, context: Dict[str, Any] = None, schema_names: List[str] = None) -> Dict[str, Any]:
        """Extract metadata from document text."""
        return extract_metadata(text, context, schema_names)

    @staticmethod
    def create_extractor(schema_names: List[str] = None) -> MetadataExtractor:
        """Create a metadata extractor for specified schemas."""
        return create_extractor(schema_names)

    @staticmethod
    def load_schema(name: str, yaml_path: Union[str, Path]) -> None:
        """Load a metadata schema from YAML file.

        Args:
            name: Name to register the schema under
            yaml_path: Path to YAML schema definition

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid or missing required fields
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Schema file not found: {yaml_path}")

        try:
            with yaml_path.open('r') as f:
                schema_dict = yaml.safe_load(f)

            if not isinstance(schema_dict, dict):
                raise ValueError("Schema YAML must define an object")

            schema = MetadataSchema(
                name=name,
                description=schema_dict.get('description', '')
            )

            # Process field definitions
            for field_name, field_def in schema_dict.get('properties', {}).items():
                field_type_str = field_def.get('type', 'string').lower()
                try:
                    field_type = MetadataFieldType(field_type_str)
                except ValueError:
                    logger.warning(f"Invalid field type '{field_type_str}' for field '{field_name}', defaulting to string")
                    field_type = MetadataFieldType.STRING

                field = MetadataField(
                    name=field_name,
                    field_type=field_type,
                    description=field_def.get('description', ''),
                    required=field_name in schema_dict.get('required', []),
                    default=field_def.get('default'),
                    pattern=field_def.get('pattern')
                )
                schema.add_field(field)

            register_schema(schema)
            logger.info(f"Loaded schema '{name}' from {yaml_path}")

        except Exception as e:
            raise ValueError(f"Failed to load schema from {yaml_path}: {e}")

