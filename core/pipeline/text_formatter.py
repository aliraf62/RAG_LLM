"""
core.pipeline.text_formatter
==========================

Registry and interfaces for pluggable text formatters used across extractors.

Text formatters convert structured data into human-readable text
representations based on configurable templates and rules.

This module:
1. Defines the base TextFormatter interface
2. Provides standard formatters (template, default)
3. Allows registration of custom formatters
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json
import logging
import re

from core.utils.component_registry import register

logger = logging.getLogger(__name__)


class BaseTextFormatter(ABC):
    """Base abstract class for text formatters."""

    @abstractmethod
    def format(self, entity_name: str, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Format entity data as text.

        Args:
            entity_name: Name of the entity being formatted
            data: Entity data dictionary
            context: Additional context information (optional)

        Returns:
            Formatted text
        """
        pass


@register("text_formatter", "default")
class DefaultTextFormatter(BaseTextFormatter):
    """Default text formatter with standard representation."""

    def __init__(self, **config):
        """Initialize formatter with configuration parameters."""
        self.config = config
        self.max_value_length = config.get("max_value_length", 500)
        self.include_empty = config.get("include_empty", False)

    def format(self, entity_name: str, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Format entity data with standard representation."""
        context = context or {}

        # Start with entity name as a heading
        text = f"# {entity_name.replace('_', ' ').title()}\n\n"

        # Process content field if specified
        content_field = context.get("content_field")
        if content_field and content_field in data:
            content = data[content_field]
            if content:
                # Handle JSON content
                if isinstance(content, str) and (content.startswith("{") or content.startswith("[")):
                    try:
                        content_obj = json.loads(content)
                        if isinstance(content_obj, dict) and "text" in content_obj:
                            text += str(content_obj["text"]) + "\n\n"
                        else:
                            # Add formatted JSON
                            text += json.dumps(content_obj, indent=2) + "\n\n"
                    except json.JSONDecodeError:
                        # Not JSON, add as is
                        text += str(content) + "\n\n"
                else:
                    # Add content as is
                    text += str(content) + "\n\n"

        # Create a table for other fields
        text += "## Details\n\n"
        text += "| Field | Value |\n"
        text += "| ----- | ----- |\n"

        # Skip fields that should be excluded
        exclude_fields = set(context.get("exclude_fields", []))
        if content_field:
            exclude_fields.add(content_field)

        # Add each field
        for key, value in sorted(data.items()):
            if key in exclude_fields:
                continue

            # Skip empty values if configured
            if not self.include_empty and (value is None or value == ""):
                continue

            # Format the value
            if value is None:
                formatted_value = ""
            elif isinstance(value, (dict, list)):
                try:
                    formatted_value = json.dumps(value, indent=2)
                except:
                    formatted_value = str(value)
            elif isinstance(value, str) and len(value) > self.max_value_length:
                formatted_value = value[:self.max_value_length] + "..."
            else:
                formatted_value = str(value)

            text += f"| {key} | {formatted_value} |\n"

        return text


@register("text_formatter", "template")
class TemplateTextFormatter(BaseTextFormatter):
    """Template-based text formatter."""

    def __init__(self, template: str = "{entity_name}: {summary}", **config):
        """Initialize formatter with a template string."""
        self.template = template
        self.config = config

    def format(self, entity_name: str, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Format entity data using template substitution."""
        context = context or {}

        # Create a dict with base values and all data fields
        values = {
            "entity_name": entity_name,
            **data
        }

        # Add summary if not already provided
        if "summary" not in values:
            values["summary"] = self._create_summary(data)

        # Add index placeholder for numbered items
        if "index" not in values and context.get("index") is not None:
            values["index"] = context.get("index")

        # Support for conditional content with | separator
        # Format: {field|default} will use default if field is empty
        def replace_with_fallback(match):
            parts = match.group(1).split('|')
            if len(parts) == 1:
                key = parts[0]
                if key in values and values[key]:
                    return str(values[key])
                return ""
            else:
                key, default = parts[0], parts[1]
                if key in values and values[key] and str(values[key]).lower() != "nan":
                    return str(values[key])
                return default

        # Replace placeholders in template
        text = self.template
        text = re.sub(r'\{([^{}]+)\}', replace_with_fallback, text)

        return text

    def _create_summary(self, data: Dict[str, Any]) -> str:
        """Create a summary from data fields."""
        # Use the first field that could be a title
        title_candidates = ["name", "title", "subject", "description"]
        for field in title_candidates:
            if field in data and data[field] and str(data[field]).lower() != "nan":
                return str(data[field])

        # Fallback to a list of key-value pairs
        summary_parts = []
        for key, value in list(data.items())[:3]:  # Just first 3 fields
            if value is not None and value != "" and str(value).lower() != "nan":
                summary_parts.append(f"{key}: {value}")

        return ", ".join(summary_parts)


def get_formatter(formatter_config: Dict[str, Any]) -> BaseTextFormatter:
    """Create a text formatter instance from configuration."""
    from core.utils.component_registry import get

    # Get formatter type
    formatter_type = formatter_config.get("type", "default")

    # If type is "custom_function", it's a special case handled by the extractor
    if formatter_type == "custom_function":
        return None

    # Check if formatter exists in registry
    try:
        formatter_class = get("text_formatter", formatter_type)
        # Pass remaining config as kwargs
        config = {k: v for k, v in formatter_config.items() if k != "type"}
        return formatter_class(**config)
    except Exception as e:
        logger.warning(f"Failed to create formatter of type '{formatter_type}': {e}")
        # Fall back to default formatter
        return DefaultTextFormatter()
