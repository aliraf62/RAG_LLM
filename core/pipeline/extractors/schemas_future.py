"""
Excel relationship schema definitions.

This module provides schemas and configs to define relational data structures
in Excel files, enabling the generic ExcelExtractor to process any structured Excel
without requiring custom code for each format.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from enum import Enum, auto
from pydantic import BaseModel, Field

class RelationType(str, Enum):
    """Types of relationships between Excel tables."""
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    ONE_TO_ONE = "one_to_one"
    MANY_TO_MANY = "many_to_many"
    SELF_REFERENCE = "self_reference"  # For hierarchical/tree data

class TableConfig(BaseModel):
    """Configuration for a single table/sheet in an Excel file."""
    # Core properties
    name: str = Field(..., description="Logical name for this table")
    sheet_name: str = Field(..., description="Name of the Excel sheet containing this table")
    header_row: int = Field(0, description="Row number (0-based) to use as the header")

    # Primary key
    primary_key: Optional[str] = Field(None, description="Column name to use as primary key")

    # Output formatting
    title_column: Optional[str] = Field(None, description="Column to use as the title in the output")
    content_column: Optional[str] = Field(None, description="Column to use as the main content")
    text_prefix: Optional[str] = Field(None, description="Prefix to add to the text output (e.g. 'Guide: ')")

    # Custom parsing
    json_columns: List[str] = Field(default_factory=list, description="Columns containing JSON to be parsed")

    # Asset handling
    asset_columns: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of column names to asset types (e.g. 'image_col': 'image')"
    )

    # Type mapping and formatting
    column_types: Dict[str, str] = Field(
        default_factory=dict,
        description="Map columns to specific types (text, number, json, etc.)"
    )

    # Advanced options
    filter_criteria: Optional[Dict[str, Any]] = Field(
        None,
        description="Filter criteria to apply when loading the table"
    )

    # Additional load parameters for pandas
    load_kwargs: Dict[str, Any] = Field(default_factory=dict)

class RelationshipConfig(BaseModel):
    """Configuration for relationships between tables."""
    # Core relationship properties
    source_table: str = Field(..., description="Name of the source table")
    target_table: str = Field(..., description="Name of the target table")
    type: RelationType = Field(..., description="Type of relationship")

    # Key columns for the relationship
    source_key: str = Field(..., description="Column in source table for the relationship")
    target_key: str = Field(..., description="Column in target table for the relationship")

    # Output structure configuration
    target_property_name: str = Field(
        ...,
        description="Name to use for the target objects in the output structure"
    )
    include_in_text: bool = Field(
        True,
        description="Whether to include the target objects in the flattened text"
    )

    # Text formatting for the relationship
    text_separator: str = Field("\n\n", description="Separator between parent and child text")
    child_text_format: str = Field(
        "{text}",
        description="Format string for child text (e.g., 'Step: {text}')"
    )

class AssetHandlingConfig(BaseModel):
    """Configuration for how to handle asset references in the data."""
    enabled: bool = Field(True, description="Whether to process assets")
    copy_assets: bool = Field(True, description="Whether to copy assets to output directory")
    asset_base_dir: Optional[str] = Field(None, description="Base directory for assets")
    caption_images: bool = Field(False, description="Whether to generate captions for images")
    asset_column_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of table.column to asset type"
    )

class WorkflowConfig(BaseModel):
    """
    Configuration for workflow-specific extraction patterns.

    This defines how data should be extracted and processed
    for a specific type of Excel file structure.
    """
    # Core properties
    name: str = Field(..., description="Name of this workflow configuration")
    version: str = Field("1.0", description="Version of this workflow configuration")
    description: Optional[str] = Field(None, description="Description of this workflow")

    # Schema definition (can be inline or referenced)
    schema: Optional[Dict[str, Any]] = Field(None, description="Schema definition or reference")
    schema_name: Optional[str] = Field(None, description="Name of schema to use from settings")

    # Workflow handlers
    handlers: Optional[Dict[str, str]] = Field(
        None,
        description="Map of handler names to import paths (e.g. 'table1': 'core.workflows.handlers.process_table1')"
    )

    # Processing options
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional options for workflow processing"
    )

    # Output configuration
    output_format: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration for output format"
    )

    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow extra attributes to be stored

class ExcelRelationalSchema(BaseModel):
    """Complete schema for defining an Excel relational structure."""
    # Schema identification
    schema_name: str = Field(..., description="Name of this schema")
    schema_version: str = Field("1.0", description="Version of the schema")
    description: Optional[str] = Field(None, description="Description of this schema")

    # Tables in the schema
    tables: Dict[str, TableConfig] = Field(..., description="Definitions of tables")

    # Relationships between tables
    relationships: List[RelationshipConfig] = Field(
        default_factory=list,
        description="Relationships between tables"
    )

    # Asset handling configuration
    asset_config: AssetHandlingConfig = Field(
        default_factory=AssetHandlingConfig,
        description="Asset handling configuration"
    )

    # Output format configuration
    output_format: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for output formatting"
    )

# Example CSO Workflow schema (for reference/testing)
CSO_WORKFLOW_SCHEMA = ExcelRelationalSchema(
    schema_name="cso_workflow",
    description="Schema for CSO workflow guides",
    tables={
        "guide": TableConfig(
            name="guide",
            sheet_name="guide",
            primary_key="Id",
            title_column="Name",
            text_prefix="Guide: ",
        ),
        "step": TableConfig(
            name="step",
            sheet_name="guideStep",
            primary_key="Id",
            title_column="Title",
            text_prefix="Step: ",
        ),
        "section": TableConfig(
            name="section",
            sheet_name="guideStepSection",
            primary_key="Id",
            text_prefix="Section: ",
            json_columns=["Content"],
        ),
        "doc": TableConfig(
            name="doc",
            sheet_name="guideDoc",
            asset_columns={"DocName": "document"},
        ),
    },
    relationships=[
        RelationshipConfig(
            source_table="guide",
            target_table="step",
            type=RelationType.ONE_TO_MANY,
            source_key="Id",
            target_key="GuideId",
            target_property_name="steps",
            child_text_format="Step: {title}\n\n{text}",
        ),
        RelationshipConfig(
            source_table="step",
            target_table="section",
            type=RelationType.ONE_TO_MANY,
            source_key="Id",
            target_key="StepId",
            target_property_name="sections",
            child_text_format="Section: {Title}\n{Text}",
        ),
        RelationshipConfig(
            source_table="section",
            target_table="section",
            type=RelationType.SELF_REFERENCE,
            source_key="Id",
            target_key="ParentSectionId",
            target_property_name="child_sections",
            child_text_format="{Text}",
        ),
        RelationshipConfig(
            source_table="section",
            target_table="doc",
            type=RelationType.ONE_TO_MANY,
            source_key="Id",
            target_key="SectionId",
            target_property_name="documents",
            include_in_text=False,
        ),
    ],
    asset_config=AssetHandlingConfig(
        enabled=True,
        copy_assets=True,
        caption_images=True,
        asset_column_map={
            "doc.DocName": "document",
            "section.Content.image": "image",  # For JSON content with image field
        },
    ),
)
