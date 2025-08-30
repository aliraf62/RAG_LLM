"""
core/pipeline/extractors/excel_utils.py

Utility functions for working with Excel data across different extractors.
Provides common functionality that can be reused by both base and customer-specific
Excel extractors.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from uuid import uuid4

import pandas as pd

from core.services.asset_manager import asset_manager
from core.utils.exceptions import AssetProcessingError

logger = logging.getLogger(__name__)


def clean_column_names(columns: List[Any]) -> List[Any]:
    """
    Clean Excel column names by removing angle brackets and special formatting.

    Parameters
    ----------
    columns : list
        List of column names to clean

    Returns
    -------
    list
        List of cleaned column names
    """
    def clean_col(col: Any) -> Any:
        if isinstance(col, str):
            col = col.strip("<>")
            if "|" in col:
                col = col.split("|")[-1]
            return col
        return col

    return [clean_col(c) for c in columns]


def process_assets(
    asset_paths: List[str],
    customer_id: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Process asset paths, copying them to the customer directory if needed.

    Parameters
    ----------
    asset_paths : List[str]
        List of asset paths to process
    customer_id : str, optional
        Customer identifier for asset storage
    settings : Dict[str, Any], optional
        Settings dictionary for configuration options

    Returns
    -------
    List[str]
        List of processed asset paths

    Raises
    ------
    AssetProcessingError
        If asset processing fails and strict mode is enabled
    """
    settings = settings or {}
    processed_assets = []

    for asset_path in asset_paths:
        if not asset_path:
            continue

        try:
            # Copy the asset to the customer directory
            processed_path = asset_manager.copy_asset(
                source_path=asset_path,
                customer_id=customer_id,
                ensure_unique=True
            )

            # Generate captions if enabled
            if settings.get("CAPTION_ASSETS", False):
                caption = asset_manager.caption_image(processed_path)
                if caption:
                    logger.debug(f"Generated caption for {processed_path}: {caption}")

            processed_assets.append(str(processed_path))

        except Exception as e:
            logger.warning(f"Failed to process asset {asset_path}: {e}")
            if settings.get("STRICT_ASSET_PROCESSING", False):
                raise AssetProcessingError(f"Asset processing failed for {asset_path}") from e

    return processed_assets


def prepare_dataframes(
    data: Union[Dict[str, 'pd.DataFrame'], List[Any]],
    field_renames: Optional[Dict[str, str]] = None
) -> Dict[str, 'pd.DataFrame']:
    """
    Prepare DataFrames by cleaning column names and applying renames.
    Accepts either a dict of DataFrames (legacy) or a list of Row objects (canonical pipeline).

    Parameters
    ----------
    data : Dict[str, pd.DataFrame] or List[Row]
        Dictionary of DataFrames or list of Row objects
    field_renames : Dict[str, str], optional
        Dictionary mapping old column names to new column names

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of cleaned DataFrames
    """
    cleaned_data = {}
    field_renames = field_renames or {}

    if isinstance(data, dict):
        for sheet_name, df in data.items():
            df.columns = clean_column_names(df.columns)
            df = df.rename(columns=field_renames)
            cleaned_data[sheet_name] = df
        return cleaned_data
    elif isinstance(data, list) and data and type(data[0]).__name__ == "Row":
        import pandas as pd
        # Try to group by sheet_name and reconstruct DataFrames
        sheets = {}
        for row in data:
            sheet = row.metadata.get('sheet_name', 'Sheet1')
            columns = row.metadata.get('columns')
            values = row.metadata.get('values')  # or whatever key holds the row's data
            if columns and values:
                if sheet not in sheets:
                    sheets[sheet] = {'columns': columns, 'rows': []}
                sheets[sheet]['rows'].append(values)
        for sheet, content in sheets.items():
            df = pd.DataFrame(content['rows'], columns=content['columns'])
            df.columns = clean_column_names(df.columns)
            df = df.rename(columns=field_renames or {})
            cleaned_data[sheet] = df
        if not cleaned_data:
            # fallback: treat as generic DataFrame
            cleaned_data["Rows"] = pd.DataFrame([row.__dict__ for row in data])
        return cleaned_data
    else:
        raise ValueError(f"prepare_dataframes: Unsupported data type: {type(data)}")


def create_relation_maps(
    dataframes: Dict[str, pd.DataFrame],
    relation_configs: Dict[str, Dict[str, str]]
) -> Dict[str, pd.core.groupby.DataFrameGroupBy]:
    """
    Create lookup maps for relational data based on foreign keys.

    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary of DataFrames
    relation_configs : Dict[str, Dict[str, str]]
        Configuration for relations between DataFrames
        Format: {"relation_name": {"table": "table_name", "key": "key_column"}}

    Returns
    -------
    Dict[str, pd.core.groupby.DataFrameGroupBy]
        Dictionary of groupby objects for lookups
    """
    relation_maps = {}

    for relation_name, config in relation_configs.items():
        table_name = config.get("table")
        key_column = config.get("key")

        if table_name in dataframes and key_column:
            df = dataframes[table_name]
            if key_column in df.columns:
                relation_maps[relation_name] = df.groupby(key_column)
            else:
                logger.warning(f"Key column '{key_column}' not found in table '{table_name}'")

    return relation_maps


def generate_row_id(
    prefix: str,
    file_path: Optional[Union[str, Path]] = None,
    identifiers: Optional[List[Any]] = None
) -> str:
    """
    Generate a unique ID for a row.

    Parameters
    ----------
    prefix : str
        Prefix for the ID
    file_path : str | Path, optional
        Path to the source file
    identifiers : list, optional
        List of identifier values to include in the ID

    Returns
    -------
    str
        Unique ID for the row
    """
    # Start with the prefix
    id_parts = [prefix]

    # Add the file name without extension if available
    if file_path:
        path = Path(file_path) if isinstance(file_path, str) else file_path
        id_parts.append(path.stem)

    # Add any provided identifiers
    if identifiers:
        id_parts.extend([str(i) for i in identifiers if i is not None])

    # Add a unique UUID segment
    id_parts.append(uuid4().hex[:8])

    # Join with underscores
    return "_".join(id_parts)


def format_structured_text(
    data: Dict[str, Any],
    format_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format structured data as readable text.

    Parameters
    ----------
    data : Dict[str, Any]
        Structured data to format
    format_config : Dict[str, Any], optional
        Configuration for formatting

    Returns
    -------
    str
        Formatted text
    """
    format_config = format_config or {}
    include_fields = format_config.get("include_fields")
    exclude_fields: Set[str] = set(format_config.get("exclude_fields", []))
    field_order = format_config.get("field_order", [])

    lines = []

    # Add title if present
    if "title" in data and data["title"]:
        title_prefix = format_config.get("title_prefix", "")
        lines.append(f"{title_prefix}{data['title']}")

    # Process fields in order if specified
    processed_fields: Set[str] = set()
    for field in field_order:
        if field in data and (include_fields is None or field in include_fields) and field not in exclude_fields:
            field_value = data[field]
            if field_value:
                lines.append(f"{field}: {field_value}")
            processed_fields.add(field)

    # Process remaining fields
    for field, value in data.items():
        if field in processed_fields or field == "title":
            continue
        if value and (include_fields is None or field in include_fields) and field not in exclude_fields:
            lines.append(f"{field}: {value}")

    return "\n".join(lines)


def dataframe_to_text(
    df: pd.DataFrame,
    sheet_name: Optional[str] = None,
    max_rows: int = 100,
    max_cols: int = 20
) -> str:
    """
    Convert a DataFrame to a readable text representation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert
    sheet_name : str, optional
        Name of the sheet for header information
    max_rows : int, optional
        Maximum number of rows to include
    max_cols : int, optional
        Maximum number of columns to include

    Returns
    -------
    str
        Textual representation of the DataFrame
    """
    if df.empty:
        return f"Sheet '{sheet_name or 'Unknown'}' is empty."

    # Header for this sheet
    header_parts = []
    if sheet_name:
        header_parts.append(f"Sheet: {sheet_name}")

    header_parts.append(f"Columns: {', '.join(str(col) for col in df.columns)}")
    header_parts.append(f"Rows: {len(df)}")
    header = "\n".join(header_parts) + "\n\n"

    # Convert to string with reasonable formatting
    try:
        # Limit to first max_rows rows to prevent extremely large texts
        sample = df.head(max_rows) if len(df) > max_rows else df
        table_str = sample.to_string(index=False, max_rows=max_rows, max_cols=max_cols)

        # Add a note if we're showing a sample
        if len(df) > max_rows:
            table_str += f"\n\n[Note: Showing first {max_rows} rows out of {len(df)} total rows]"

        return header + table_str
    except Exception as e:
        logger.warning(f"Error converting DataFrame to string: {e}")
        # Fallback: Convert each row to a simple string representation
        rows_text = [", ".join(str(val) for val in row) for _, row in sample.iterrows()]
        return header + "\n".join(rows_text)
