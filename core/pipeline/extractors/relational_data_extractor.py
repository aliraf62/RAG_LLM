"""
Relational Data Extractor for handling complex data relationships.

This extractor processes data from relational sources (Excel, SQL, CSV) and
handles complex relationships between entities based on a configuration schema.
It transforms raw relational data into structured Row objects for downstream
processing in the pipeline.

Key features:
- Configuration-driven relationship processing
- Support for hierarchical relationships
- Asset reference resolution
- Metadata extraction and normalization
- Compatible with various relational data sources
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import json
import logging
import re
import yaml

import pandas as pd

from core.pipeline.base import Row
from core.pipeline.extractors.base import BaseExtractor
from core.pipeline.extractors.excel_utils import (
    clean_column_names, generate_row_id
)
from core.utils.component_registry import register
from core.services.asset_manager import copy_asset
from core.pipeline.utils.image_captioner import caption_image
from core.metadata.processor import process_document_metadata
from core.config.settings import settings

logger = logging.getLogger(__name__)


@register("extractor", "relational_data")
class RelationalDataExtractor(BaseExtractor):
    """
    Generic extractor for relational data from various sources (Excel, SQL, CSV).

    This extractor processes complex relationships between tables/entities
    based on a configuration schema, transforming raw data into structured
    Row objects for downstream processing.

    The extractor handles:
    - Entity relationships (one-to-one, one-to-many, many-to-many)
    - Hierarchical structures (parent-child relationships)
    - Asset references and resolution
    - Metadata extraction and normalization
    """

    def __init__(
        self,
        customer_id: Optional[str] = None,
        schema_config: Optional[Union[dict, str, Path]] = None,
        clean_columns: bool = True,
        extract_assets: bool = True,
        **config: Any
    ) -> None:
        """
        Initialize the relational data extractor.

        Parameters
        ----------
        customer_id : str, optional
            Customer identifier for metadata processing.
        schema_config : dict, str, or Path, optional
            Configuration schema defining entity relationships.
            Can be a dictionary, path to YAML file, or YAML string.
        clean_columns : bool, optional
            Whether to clean column names (remove special chars, etc.)
        extract_assets : bool, optional
            Whether to extract assets from appropriate columns
        **config : dict
            Additional configuration parameters.
        """
        super().__init__(**config)
        self.customer_id = customer_id or config.get("customer_id")
        self.clean_columns = clean_columns
        self.extract_assets = extract_assets

        # Load the schema configuration
        self.schema = self._load_schema_config(schema_config or config.get("schema_config", {}))

        # File path for asset resolution
        self.file_path = config.get("file_path")
        self.assets_dir = config.get("assets_dir")

        # Cache for processed entities to avoid duplicate processing
        self._processed_entities = set()

        # Format configuration for text representation
        self.format_config = config.get("format_config", {
            "include_columns": True,
            "max_value_length": 500,
            "include_empty": False
        })

    def _load_schema_config(self, schema_config: Union[dict, str, Path]) -> dict:
        """
        Load and validate the schema configuration.

        Parameters
        ----------
        schema_config : dict, str, or Path
            Configuration schema. Can be a dictionary, path to YAML file,
            or YAML string.

        Returns
        -------
        dict
            Validated schema configuration.
        """
        if isinstance(schema_config, (str, Path)):
            # Check if it's a file path
            path = Path(schema_config)
            if path.exists() and path.is_file():
                with open(path, "r") as f:
                    schema = yaml.safe_load(f)
            else:
                # Assume it's a YAML string
                schema = yaml.safe_load(schema_config)
        else:
            # Assume it's already a dictionary
            schema = schema_config

        # Validate schema structure
        if not isinstance(schema, dict):
            raise ValueError("Schema configuration must be a dictionary")

        # Ensure minimum required keys
        required_keys = ["entities", "relationships"]
        missing_keys = [key for key in required_keys if key not in schema]
        if missing_keys:
            raise ValueError(f"Schema configuration missing required keys: {missing_keys}")

        return schema

    def extract_rows(self) -> Iterable[Row]:
        """
        Extract Row objects from relational data based on schema configuration.

        This method processes input data according to the defined schema,
        resolving relationships and transforming the data into Row objects.

        Returns
        -------
        Iterable[Row]
            Row objects containing text, metadata, structured data, and assets
        """
        # Reset the processed entities cache for a fresh extraction
        self._processed_entities = set()

        # Get the input rows from the loader
        input_rows = self.config.get("rows", [])
        if not input_rows:
            logger.warning("No input rows provided by the loader")
            return

        # Prepare dataframes from input rows
        dataframes = self._prepare_dataframes(input_rows)
        if not dataframes:
            logger.warning("No dataframes found in input rows")
            return

        # Get export limit if configured
        export_limit = self.config.get("export_limit")
        if export_limit is not None:
            try:
                export_limit = int(export_limit)
                logger.info(f"Limiting extraction to {export_limit} entities per type")
            except (ValueError, TypeError):
                logger.warning(f"Invalid export_limit value: {export_limit}, ignoring limit")
                export_limit = None

        # Process root entities first (those that don't depend on others)
        root_entities = self._get_root_entities()
        entity_counts = {entity: 0 for entity in root_entities}

        for entity_name in root_entities:
            if entity_name not in dataframes:
                logger.warning(f"Root entity '{entity_name}' not found in dataframes")
                continue

            # Get entity configuration
            entity_config = self.schema["entities"].get(entity_name, {})
            df = dataframes[entity_name]

            # Process each row of the root entity
            for idx, row in df.iterrows():
                # Check if we've reached the export limit for this entity type
                if export_limit is not None and entity_counts.get(entity_name, 0) >= export_limit:
                    logger.info(f"Export limit reached for entity type: {entity_name}")
                    break

                # Generate a unique ID for this entity row
                entity_id = row.get(entity_config.get("primary_key", "id"))
                if entity_id is None:
                    entity_id = idx

                # Skip if already processed
                entity_cache_key = f"{entity_name}:{entity_id}"
                if entity_cache_key in self._processed_entities:
                    continue
                self._processed_entities.add(entity_cache_key)

                # Process this entity and its relationships
                for item in self._process_entity(entity_name, entity_id, row, dataframes):
                    entity_counts[entity_name] = entity_counts.get(entity_name, 0) + 1
                    yield item

    def _prepare_dataframes(self, input_rows: List[Row]) -> Dict[str, pd.DataFrame]:
        """
        Prepare dataframes from input rows.

        Parameters
        ----------
        input_rows : List[Row]
            Input rows from the loader with structured dataframe data.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping entity names to dataframes.
        """
        dataframes = {}

        for row in input_rows:
            # Skip rows without structured data
            if not row.structured or "dataframe" not in row.structured:
                continue

            # Get the dataframe and sheet info
            df_records = row.structured.get("dataframe", [])
            df = pd.DataFrame(df_records)

            if df.empty:
                continue

            # Clean column names if configured
            if self.clean_columns:
                df.columns = clean_column_names(df.columns)

            # Get the sheet name or logical name
            sheet_name = row.structured.get("sheet_name", row.metadata.get("sheet_name", "unknown"))
            logical_name = row.metadata.get("logical_name", sheet_name)

            # Map the sheet name to the entity name using schema
            entity_name = self._map_sheet_to_entity(sheet_name, logical_name)

            # Store the dataframe with the entity name
            dataframes[entity_name] = df

        return dataframes

    def _map_sheet_to_entity(self, sheet_name: str, logical_name: str) -> str:
        """
        Map sheet name to entity name using schema configuration.

        Parameters
        ----------
        sheet_name : str
            Original sheet name.
        logical_name : str
            Logical name of the sheet.

        Returns
        -------
        str
            Entity name mapped from sheet name.
        """
        # Get sheet mapping from schema
        sheet_mapping = self.schema.get("sheet_mapping", {})

        # Check if there's a direct mapping
        if sheet_name in sheet_mapping:
            return sheet_mapping[sheet_name]
        if logical_name in sheet_mapping:
            return sheet_mapping[logical_name]

        # Check for pattern-based mapping
        for pattern, entity_name in sheet_mapping.items():
            if pattern.startswith("regex:"):
                regex = pattern[6:]
                if re.search(regex, sheet_name) or re.search(regex, logical_name):
                    return entity_name

        # Default to sheet name if no mapping found
        return sheet_name

    def _get_root_entities(self) -> List[str]:
        """
        Get root entities that don't depend on other entities.

        Returns
        -------
        List[str]
            List of root entity names.
        """
        # Get all entities
        all_entities = set(self.schema["entities"].keys())

        # Get entities that are referenced as children in relationships
        child_entities = set()
        for rel in self.schema.get("relationships", []):
            child_entity = rel.get("child")
            if child_entity:
                child_entities.add(child_entity)

        # Root entities are those that are not children
        root_entities = all_entities - child_entities

        # If no root entities found, default to all entities
        if not root_entities:
            logger.warning("No root entities found, using all entities")
            root_entities = all_entities

        return list(root_entities)

    def _process_entity(
        self,
        entity_name: str,
        entity_id: Any,
        row_data: pd.Series,
        dataframes: Dict[str, pd.DataFrame]
    ) -> Iterable[Row]:
        """
        Process an entity and its relationships, yielding Row objects.

        Parameters
        ----------
        entity_name : str
            Name of the entity to process.
        entity_id : Any
            ID of the entity row.
        row_data : pd.Series
            Data for this entity row.
        dataframes : Dict[str, pd.DataFrame]
            All available dataframes.

        Returns
        -------
        Iterable[Row]
            Row objects for this entity and its related entities.
        """
        # Convert row to dictionary
        row_dict = row_data.to_dict()

        # Get entity configuration
        entity_config = self.schema["entities"].get(entity_name, {})
        primary_key = entity_config.get("primary_key", "id")

        # Create a readable text representation
        text_content = self._format_entity_text(entity_name, row_dict)

        # Generate a unique ID for this row
        file_path = self.file_path or "unknown"
        row_id = generate_row_id("relational", file_path, [entity_name, entity_id])

        # Process child relationships
        structured_data = self._process_relationships(
            entity_name, entity_id, row_dict, dataframes
        )

        # Add the base entity data to structured
        structured_data["entity"] = {
            "name": entity_name,
            "id": entity_id,
            "data": row_dict
        }

        # Basic metadata
        base_metadata = {
            "source": file_path,
            "source_type": "relational_data",
            "entity_name": entity_name,
            "entity_id": entity_id
        }

        # Add fields to metadata based on schema
        metadata_fields = entity_config.get("metadata_fields", [])
        metadata = {**base_metadata}
        for field in metadata_fields:
            if field in row_dict:
                metadata[field] = row_dict[field]

        # Process metadata using the metadata processing system
        processed_metadata = self._process_metadata(metadata, text_content)

        # Extract assets if configured
        assets = []
        if self.extract_assets:
            assets = self._extract_assets(entity_name, row_dict, structured_data)

        # Create and yield a Row object for this entity
        yield Row(
            text=text_content,
            metadata=processed_metadata,
            structured=structured_data,
            assets=assets,
            id=row_id
        )

    def _process_relationships(
        self,
        entity_name: str,
        entity_id: Any,
        row_data: dict,
        dataframes: Dict[str, pd.DataFrame]
    ) -> dict:
        """
        Process relationships for an entity, building structured data.

        Parameters
        ----------
        entity_name : str
            Name of the entity.
        entity_id : Any
            ID of the entity row.
        row_data : dict
            Data for this entity row.
        dataframes : Dict[str, pd.DataFrame]
            All available dataframes.

        Returns
        -------
        dict
            Structured data with resolved relationships.
        """
        structured_data = {"relationships": {}}

        # Get all relationships where this entity is the parent
        for rel in self.schema.get("relationships", []):
            if rel.get("parent") != entity_name:
                continue

            # Get relationship details
            child_entity = rel.get("child")
            rel_type = rel.get("type", "one_to_many")
            parent_key = rel.get("via_fields", {}).get("parent_key")
            child_key = rel.get("via_fields", {}).get("child_key")

            # Skip if missing required fields
            if not (child_entity and parent_key and child_key):
                logger.warning(f"Incomplete relationship definition for {entity_name}->{child_entity}")
                continue

            # Skip if child dataframe not available
            if child_entity not in dataframes:
                logger.warning(f"Child entity '{child_entity}' not found in dataframes")
                continue

            # Get the parent ID value
            parent_id_value = row_data.get(parent_key)
            if parent_id_value is None:
                logger.warning(f"Parent key '{parent_key}' not found in {entity_name} row")
                continue

            # Filter child dataframe for matching rows
            child_df = dataframes[child_entity]
            if child_key not in child_df.columns:
                logger.warning(f"Child key '{child_key}' not found in {child_entity} dataframe")
                continue

            # Match child rows
            child_rows = child_df[child_df[child_key] == parent_id_value]

            # Process each child row
            child_data = []
            for idx, child_row in child_rows.iterrows():
                # Convert to dict
                child_dict = child_row.to_dict()

                # Get the child entity configuration
                child_config = self.schema["entities"].get(child_entity, {})
                child_primary_key = child_config.get("primary_key", "id")
                child_id = child_dict.get(child_primary_key, idx)

                # Mark this child as processed
                child_cache_key = f"{child_entity}:{child_id}"
                if child_cache_key in self._processed_entities:
                    continue
                self._processed_entities.add(child_cache_key)

                # Process the child entity
                child_data.append({
                    "id": child_id,
                    "data": child_dict
                })

            # Add relationship data to structured
            structured_data["relationships"][child_entity] = {
                "type": rel_type,
                "data": child_data
            }

        return structured_data

    def _format_entity_text(self, entity_name: str, row_data: dict) -> str:
        """
        Format entity data as readable text.

        Parameters
        ----------
        entity_name : str
            Name of the entity.
        row_data : dict
            Entity row data.

        Returns
        -------
        str
            Formatted text representation.
        """
        # Get entity configuration
        entity_config = self.schema["entities"].get(entity_name, {})

        # Get text formatter configuration
        text_formatter = entity_config.get("text_formatter", {})
        formatter_type = text_formatter.get("type", "default")

        if formatter_type == "template":
            # Use template formatting
            template = text_formatter.get("template", "{entity_name}: {summary}")

            # Replace placeholders
            text = template.format(
                entity_name=entity_name,
                summary=self._summarize_row_data(row_dict=row_data),
                **row_data
            )
        else:
            # Default formatting
            text = self._default_text_formatter(entity_name, row_data)

        return text

    def _default_text_formatter(self, entity_name: str, row_data: dict) -> str:
        """
        Default text formatter for an entity.

        Parameters
        ----------
        entity_name : str
            Name of the entity.
        row_data : dict
            Entity row data.

        Returns
        -------
        str
            Formatted text.
        """
        # Start with entity name as a heading
        text = f"# {entity_name.replace('_', ' ').title()}\n\n"

        # Process special content fields first if defined
        entity_config = self.schema["entities"].get(entity_name, {})
        content_field = entity_config.get("content_field")

        if content_field and content_field in row_data:
            content = row_data[content_field]
            if content:
                # Check if content is a JSON string
                if isinstance(content, str) and (content.startswith("{") or content.startswith("[")):
                    try:
                        content_obj = json.loads(content)
                        if isinstance(content_obj, dict) and "text" in content_obj:
                            # Extract text from JSON content
                            text += content_obj["text"] + "\n\n"
                        elif isinstance(content_obj, dict) and "content" in content_obj:
                            # Extract content field
                            text += content_obj["content"] + "\n\n"
                        else:
                            # Add formatted JSON
                            text += json.dumps(content_obj, indent=2) + "\n\n"
                    except json.JSONDecodeError:
                        # Not JSON, add as is
                        text += str(content) + "\n\n"
                else:
                    # Add content as is
                    text += str(content) + "\n\n"

        # Add other fields as a table
        text += "## Details\n\n"
        text += "| Field | Value |\n"
        text += "| ----- | ----- |\n"

        # Skip fields that should be excluded from text
        exclude_fields = set(entity_config.get("exclude_from_text", []))

        # Add each field
        for key, value in row_data.items():
            if key in exclude_fields:
                continue

            # Format the value
            formatted_value = self._format_field_value(value)
            text += f"| {key} | {formatted_value} |\n"

        return text

    def _summarize_row_data(self, row_dict: dict) -> str:
        """
        Create a summary of row data.

        Parameters
        ----------
        row_dict : dict
            Row data to summarize.

        Returns
        -------
        str
            Summarized text.
        """
        # Use the first field that could be a title
        title_candidates = ["name", "title", "subject", "description"]
        for field in title_candidates:
            if field in row_dict and row_dict[field]:
                return str(row_dict[field])

        # Fallback to a list of key-value pairs
        summary_parts = []
        for key, value in list(row_dict.items())[:3]:  # Just first 3 fields
            if value is not None and value != "":
                summary_parts.append(f"{key}: {value}")

        return ", ".join(summary_parts)

    def _format_field_value(self, value: Any) -> str:
        """
        Format a field value for text display.

        Parameters
        ----------
        value : Any
            Field value to format.

        Returns
        -------
        str
            Formatted value.
        """
        if value is None:
            return ""

        if isinstance(value, (dict, list)):
            # Format JSON
            try:
                return json.dumps(value, indent=2)
            except:
                return str(value)

        # Truncate long string values
        if isinstance(value, str) and len(value) > self.format_config.get("max_value_length", 500):
            return value[:self.format_config.get("max_value_length", 500)] + "..."

        return str(value)

    def _extract_assets(
        self,
        entity_name: str,
        row_data: dict,
        structured_data: dict
    ) -> List[str]:
        """
        Extract assets referenced in entity data.

        Parameters
        ----------
        entity_name : str
            Name of the entity.
        row_data : dict
            Entity row data.
        structured_data : dict
            Structured data for the entity.

        Returns
        -------
        List[str]
            List of asset paths.
        """
        assets = []

        # Get entity configuration
        entity_config = self.schema["entities"].get(entity_name, {})

        # Get asset fields from configuration
        asset_fields = entity_config.get("asset_fields", [])

        # Process each asset field
        for field_config in asset_fields:
            # Get field name and type
            if isinstance(field_config, str):
                # Simple field name
                field_name = field_config
                field_type = "direct"
            else:
                # Detailed configuration
                field_name = field_config.get("field")
                field_type = field_config.get("type", "direct")

            # Skip if field not found in row data
            if not field_name or field_name not in row_data:
                continue

            # Process based on field type
            if field_type == "direct":
                # Direct file path
                value = row_data[field_name]
                if value and isinstance(value, str):
                    assets.append(value)
            elif field_type == "json":
                # JSON field with asset reference
                value = row_data[field_name]
                if not value:
                    continue

                # Parse JSON if needed
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        continue

                # Extract asset from JSON
                if isinstance(value, dict):
                    json_path = field_config.get("json_path", "")
                    if json_path:
                        # Navigate JSON path
                        parts = json_path.split(".")
                        current = value
                        for part in parts:
                            if part in current:
                                current = current[part]
                            else:
                                current = None
                                break

                        # If we found a value, add it as an asset
                        if current:
                            if isinstance(current, str):
                                assets.append(current)
                            elif isinstance(current, dict) and "path" in current:
                                assets.append(current["path"])
                            elif isinstance(current, dict) and "id" in current:
                                asset_id = current["id"]
                                # Try to find asset by ID in assets directory
                                asset_path = self._find_asset_by_id(asset_id)
                                if asset_path:
                                    assets.append(asset_path)
                    else:
                        # No path specified, look for common fields
                        for key in ["path", "url", "file", "image"]:
                            if key in value and value[key]:
                                assets.append(str(value[key]))
                                break

        return assets

    def _find_asset_by_id(self, asset_id: str) -> Optional[str]:
        """
        Find an asset file by its ID in the assets directory.

        Parameters
        ----------
        asset_id : str
            ID of the asset to find.

        Returns
        -------
        Optional[str]
            Path to the asset if found, None otherwise.
        """
        if not self.assets_dir or not asset_id:
            return None

        assets_dir = Path(self.assets_dir)
        if not assets_dir.exists() or not assets_dir.is_dir():
            return None

        # Try to find asset by ID in filename
        for asset_file in assets_dir.glob("**/*"):
            if asset_file.is_file() and str(asset_id) in asset_file.name:
                return str(asset_file)

        return None

    def _process_metadata(self, metadata: dict, text: str = "") -> dict:
        """
        Process metadata using the centralized metadata framework.

        This method ensures that:
        1. Schema-based metadata rules are applied
        2. Document structure is properly formatted for the metadata processor
        3. Canonical keys from settings.CANONICAL_METADATA_KEYS are properly applied
        4. Customer-specific metadata adaptations are processed when customer_id is available

        Parameters
        ----------
        metadata : dict
            Metadata to process.
        text : str, optional
            Associated text content, by default ""

        Returns
        -------
        dict
            Processed metadata conforming to the metadata framework standards.
        """
        # Apply schema-based metadata processing
        if "metadata_processor" in self.schema:
            metadata = self._apply_metadata_rules(metadata)

        # Create a document-like structure that the metadata processor expects
        document = {
            "text": text,
            "metadata": metadata
        }

        # Process using the metadata processor from core.metadata
        if self.customer_id:
            # Pass the document (not just metadata) to follow the processor's expected interface
            processed_doc = process_document_metadata(document, self.customer_id)
            return processed_doc["metadata"]
        else:
            # If no customer_id, ensure canonical keys are present
            if hasattr(settings, 'CANONICAL_METADATA_KEYS'):
                for key in settings.CANONICAL_METADATA_KEYS:
                    if key not in metadata:
                        metadata[key] = None
            return metadata

    def _apply_metadata_rules(self, metadata: dict) -> dict:
        """
        Apply metadata processing rules from schema.

        Parameters
        ----------
        metadata : dict
            Raw metadata.

        Returns
        -------
        dict
            Processed metadata.
        """
        result = metadata.copy()

        # Get metadata rules
        metadata_rules = self.schema.get("metadata_processor", {}).get("rules", [])

        # Apply each rule
        for rule in metadata_rules:
            rule_type = rule.get("type")

            if rule_type == "map_field":
                # Map one field to another
                source = rule.get("source")
                target = rule.get("target")
                if source in result:
                    result[target] = result[source]
            elif rule_type == "combine_fields":
                # Combine multiple fields
                sources = rule.get("sources", [])
                target = rule.get("target")
                separator = rule.get("separator", " ")
                values = []
                for source in sources:
                    if source in result and result[source]:
                        values.append(str(result[source]))
                if values:
                    result[target] = separator.join(values)
            elif rule_type == "extract_json":
                # Extract value from JSON field
                source = rule.get("source")
                target = rule.get("target")
                json_path = rule.get("json_path", "")
                if source in result and result[source]:
                    value = result[source]
                    # Parse JSON if needed
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            continue

                    # Navigate JSON path
                    if json_path:
                        path_parts = json_path.split(".")
                        current = value
                        for part in path_parts:
                            if isinstance(current, dict) and part in current:
                                current = current[part]
                            else:
                                current = None
                                break
                        value = current

                    if value is not None:
                        result[target] = value
            elif rule_type == "add_categories":
                # Add categories based on folder relationships
                entity_name = result.get("entity_name")
                entity_id = result.get("entity_id")
                if entity_name == "guide" and entity_id:
                    # Look up categories from folder relationships
                    categories = self._get_guide_categories(entity_id)
                    if categories:
                        result["categories"] = categories

        return result

    def _get_guide_categories(self, guide_id: str) -> List[str]:
        """
        Get categories for a guide from folder relationships.

        Parameters
        ----------
        guide_id : str
            ID of the guide.

        Returns
        -------
        List[str]
            List of category names.
        """
        categories = []

        # Look for guideFolderRelation in the schema
        folder_relation_entity = self.schema.get("sheet_mapping", {}).get("guideFolderRelation")
        folder_entity = self.schema.get("sheet_mapping", {}).get("guideFolder")

        if not folder_relation_entity or not folder_entity:
            return categories

        # Access dataframes from config
        input_rows = self.config.get("rows", [])
        folder_relation_df = None
        folder_df = None

        # Find the right dataframes
        for row in input_rows:
            sheet_name = row.metadata.get("sheet_name")
            if not sheet_name:
                continue

            if self._map_sheet_to_entity(sheet_name, sheet_name) == folder_relation_entity:
                folder_relation_df = pd.DataFrame(row.structured.get("dataframe", []))
            elif self._map_sheet_to_entity(sheet_name, sheet_name) == folder_entity:
                folder_df = pd.DataFrame(row.structured.get("dataframe", []))

        if folder_relation_df is None or folder_df is None:
            return categories

        # Clean column names if needed
        if self.clean_columns:
            if not folder_relation_df.empty:
                folder_relation_df.columns = clean_column_names(folder_relation_df.columns)
            if not folder_df.empty:
                folder_df.columns = clean_column_names(folder_df.columns)

        # Find guide-folder relationships
        if "guideid" in folder_relation_df.columns and "folderid" in folder_relation_df.columns:
            # Find folder IDs for this guide
            folder_relations = folder_relation_df[folder_relation_df["guideid"] == guide_id]
            folder_ids = folder_relations["folderid"].tolist()

            # Find folder names
            if "id" in folder_df.columns and "name" in folder_df.columns and "category" in folder_df.columns:
                for folder_id in folder_ids:
                    folder_rows = folder_df[folder_df["id"] == folder_id]
                    for _, folder_row in folder_rows.iterrows():
                        folder_name = folder_row["name"]
                        folder_category = folder_row["category"]
                        if folder_name:
                            categories.append(str(folder_name))
                        if folder_category and folder_category != folder_name:
                            categories.append(str(folder_category))

        return list(set(categories))  # Remove duplicates
