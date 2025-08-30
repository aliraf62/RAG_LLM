"""
Generic Excel extractor for flat or relational Excel files.

- Processes DataFrames provided by the loader
- Extracts structured data and creates meaningful text representations
- Handles metadata processing and asset detection
- No customer-specific logic (no nested fields, steps, child, or asset logic)
- Customer-specific extractors should subclass this and override extract_rows()
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List, Union
import logging

import pandas as pd

from core.pipeline.base import Row
from core.pipeline.extractors.base import BaseExtractor
from core.pipeline.extractors.excel_utils import (
    clean_column_names, process_assets, generate_row_id,
    format_structured_text, dataframe_to_text, prepare_dataframes
)
from core.utils.component_registry import register
from core.metadata.processor import process_document_metadata

logger = logging.getLogger(__name__)

@register("extractor", "excel")
class ExcelExtractor(BaseExtractor):
    """
    Generic extractor for Excel workbooks (XLSX/XLSB/CSV).
    
    The extractor is responsible for:
    - Processing DataFrames from the loader
    - Applying business logic to transform the data
    - Creating meaningful text representations
    - Extracting assets and metadata
    - Generating structured data for downstream components
    """

    def __init__(
        self,
        customer_id: Optional[str] = None,
        clean_columns: bool = True,
        extract_assets: bool = True,
        **config: Any
    ) -> None:
        """
        Parameters
        ----------
        customer_id : str, optional
            Customer identifier for metadata processing.
        clean_columns : bool, optional
            Whether to clean column names (remove special chars, etc.)
        extract_assets : bool, optional
            Whether to extract assets from appropriate columns
        config : dict
            Additional config.
        """
        super().__init__(**config)
        self.customer_id = customer_id or config.get("customer_id")
        self.clean_columns = clean_columns
        self.extract_assets = extract_assets
        self.file_path = config.get("file_path")
        
        # Format configuration for text representation
        self.format_config = config.get("format_config", {
            "include_columns": True,
            "max_value_length": 500,
            "include_empty": False
        })

    def extract_rows(self) -> Iterable[Row]:
        """
        Yields Row objects from loader-provided DataFrames.
        
        This method assumes the loader has provided Row objects with DataFrames 
        in the structured field.

        Returns
        -------
        Iterable[Row]
            Row objects containing text, metadata, structured data, and assets
        """
        for input_row in self.config.get("rows", []):
            # Extract the DataFrame from the loader's structured data
            if not input_row.structured or "dataframe" not in input_row.structured:
                logger.warning(f"No DataFrame found in Row: {input_row}")
                continue
                
            # Get DataFrame and sheet info
            df_records = input_row.structured.get("dataframe", [])
            df = pd.DataFrame(df_records)
            
            if df.empty:
                logger.warning("Empty DataFrame, skipping")
                continue
                
            # Clean column names if configured
            if self.clean_columns:
                df.columns = clean_column_names(df.columns)
                
            sheet_name = input_row.metadata.get("sheet_name", "unknown")
            logical_name = input_row.metadata.get("logical_name", sheet_name)
            
            # Process each row in the DataFrame
            for idx, row in df.iterrows():
                # Convert row to dictionary
                row_dict = row.to_dict()
                
                # Create a readable text representation of the row
                row_text = self._format_row_text(row_dict, sheet_name, idx)
                
                # Generate a unique ID for this row
                file_path = input_row.metadata.get("source", self.file_path)
                row_id = generate_row_id("excel", file_path, [sheet_name, idx])
                
                # Basic metadata about the source
                base_metadata = {
                    "source": file_path,
                    "source_type": "excel",
                    "sheet_name": sheet_name,
                    "logical_name": logical_name,
                    "row_index": idx,
                    "filename": input_row.metadata.get("filename", "unknown"),
                }
                
                # Combine with row data as metadata
                combined_metadata = {**base_metadata, **row_dict}
                
                # Process metadata using the metadata processing system
                processed_metadata = self._process_metadata(row_text, combined_metadata)
                
                # Store the full row data in the structured field for advanced processing
                structured_data = {
                    "row": row_dict,
                    "sheet": {
                        "name": sheet_name,
                        "logical_name": logical_name,
                    }
                }
                
                # Get asset paths if any
                assets = []
                if self.extract_assets:
                    assets = self._extract_assets_from_row(row_dict)
                
                # Create and yield a proper Row object
                yield Row(
                    text=row_text,
                    metadata=processed_metadata,
                    structured=structured_data,
                    assets=assets,
                    id=row_id
                )

    def _format_row_text(self, row_dict: Dict[str, Any], sheet_name: str, idx: int) -> str:
        """
        Format a row dictionary into a readable text representation.
        
        Parameters
        ----------
        row_dict : Dict[str, Any]
            Dictionary of row values
        sheet_name : str
            Name of the sheet
        idx : int
            Row index
            
        Returns
        -------
        str
            Formatted text representation
        """
        # Start with sheet name and row index
        text_parts = [f"Sheet: {sheet_name}", f"Row: {idx + 1}"]
        
        # Add each column and value
        for col, val in row_dict.items():
            # Skip empty values if configured
            if val is None and not self.format_config.get("include_empty", False):
                continue
                
            # Format the value based on type
            if isinstance(val, str):
                # Truncate long string values if needed
                max_len = self.format_config.get("max_value_length", 500)
                if len(val) > max_len:
                    val_str = val[:max_len] + "..."
                else:
                    val_str = val
            else:
                val_str = str(val)
                
            text_parts.append(f"{col}: {val_str}")
            
        return "\n".join(text_parts)

    def _process_metadata(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document metadata using the metadata processing system.
        
        Parameters
        ----------
        text : str
            Document text
        metadata : Dict[str, Any]
            Raw metadata
            
        Returns
        -------
        Dict[str, Any]
            Processed metadata
        """
        try:
            # Use metadata processor if available
            processed_doc = process_document_metadata(
                {"text": text, "metadata": metadata},
                self.customer_id
            )
            return processed_doc.get("metadata", metadata)
        except Exception as e:
            logger.warning(f"Error processing metadata: {e}")
            return metadata

    def _extract_assets_from_row(self, row_data: Dict[str, Any]) -> List[str]:
        """
        Extract asset paths from row data.

        By default, looks for columns that might contain asset paths.
        Override this in subclasses for customer-specific asset handling.

        Parameters
        ----------
        row_data : Dict[str, Any]
            Row data as dictionary

        Returns
        -------
        List[str]
            List of asset paths
        """
        assets = []

        # Look for common asset column names
        asset_column_patterns = ["image", "file", "path", "asset", "doc", "attachment"]

        for col, value in row_data.items():
            col_lower = str(col).lower()
            # Check if the column name suggests it contains assets
            if any(pattern in col_lower for pattern in asset_column_patterns) and isinstance(value, str) and value:
                # Process the asset if customer_id is provided
                if self.customer_id and self.settings.get("PROCESS_ASSETS", True):
                    try:
                        processed_assets = process_assets([value], self.customer_id, self.settings)
                        assets.extend(processed_assets)
                    except Exception as e:
                        logger.warning(f"Failed to process asset from column {col}: {e}")
                else:
                    # Just add the path without processing
                    assets.append(value)

        return assets
