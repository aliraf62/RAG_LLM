"""
Excel file loader for loading Excel workbooks in various formats (XLSX/XLSB/CSV).

Handles the I/O operations of reading Excel files and provides the raw data
for extractors to process in the form of Row objects.
"""

from __future__ import annotations
from pathlib import Path
import logging
from typing import Any, Dict, Iterable, Iterator, Optional, Union, List, Set, Tuple
from uuid import uuid4

import pandas as pd

from core.config.settings import settings
from core.pipeline.base import Row
from core.pipeline.loaders.base import BaseLoader
from core.utils.component_registry import register

logger = logging.getLogger(__name__)

@register("loader", "excel")
class ExcelLoader(BaseLoader):
    """
    Loader for Excel workbooks (XLSX/XLSB/CSV).

    Provides a clean interface to read Excel files and access their content as Row objects,
    which can then be processed by other pipeline components.

    The loader is responsible for:
    - Reading the file from disk
    - Parsing the Excel format
    - Converting sheets to DataFrames
    - Creating minimal Row objects with dataframes in the structured field
    """

    def __init__(
        self,
        file_path: Optional[str | Path] = None,
        sheets: Optional[Dict[str, Any]] = None,
        sheet_load_kwargs: Optional[Dict[str, Any]] = None,
        exclude_sheets: Optional[List[str]] = None,
        customer_id: Optional[str] = None,
        **config: Any
    ) -> None:
        """
        Parameters
        ----------
        file_path : str | Path, optional
            Path to the Excel workbook.
        sheets : dict, optional
            Mapping of logical table names to sheet names and header rows.
            Example: {"guide": {"sheet_name": "guide", "header_row": 0}, ...}
        sheet_load_kwargs : dict, optional
            Extra kwargs for pd.read_excel.
        exclude_sheets : List[str], optional
            List of sheet names to exclude from loading.
        customer_id : str, optional
            Customer identifier for loading customer-specific config.
        config : dict
            Additional configuration.
        """
        super().__init__(**config)
        self.file_path = Path(file_path) if file_path else None
        self.customer_id = customer_id

        # Get customer-specific configuration
        self.customer_config = settings.get('customer_config', {})

        # Initialize sheets configuration
        self.sheets = sheets or {}

        # Get sheet load kwargs from settings if not provided
        self.sheet_load_kwargs = sheet_load_kwargs or settings.get('sheet_load_kwargs', {})
        if not self.sheet_load_kwargs:
            self.sheet_load_kwargs = {"engine": "openpyxl"}

        # Get exclude sheets configuration
        self.exclude_sheets = set(exclude_sheets or settings.get('EXCEL_LOADER_EXCLUDE_SHEETS', []))

        # Initialize DataFrame cache
        self._dataframes = {}  # Cache for loaded DataFrames
        self._sheet_names = None  # Cache for sheet names

        # Debug logging for initialization
        logger.debug(f"[ExcelLoader] Initialized with sheets: {self.sheets}, exclude_sheets: {self.exclude_sheets}")

    def _get_header_row_for_sheet(self, sheet_name: str) -> int:
        """
        Get the header row for a specific sheet from configuration.

        Parameters
        ----------
        sheet_name : str
            Name of the sheet

        Returns
        -------
        int
            Header row (0-indexed)
        """
        # Check if there's a specific configuration for this sheet
        if self.sheets is None:
            self.sheets = {}
        for logical_name, config in self.sheets.items():
            if config.get("sheet_name") == sheet_name:
                return config.get("header_row", 0)

        # Check if there's a default header row in configuration
        default_header = self.customer_config.get("loaders", {}).get("excel", {}).get("default_header_row")
        if default_header is not None:
            return default_header

        # Use the default from settings or fall back to 0
        return settings.get("EXCEL_LOADER_DEFAULT_HEADER_ROW", 0)

    def load_sheet(self, sheet_name: str, header: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Load a specific sheet from the workbook.

        Parameters
        ----------
        sheet_name : str
            Name of the sheet to load.
        header : int, optional
            Row to use as header (0-indexed). If not provided, will use configured header row.
        **kwargs : dict
            Additional arguments for pd.read_excel.

        Returns
        -------
        pd.DataFrame
            The loaded sheet as a DataFrame.
        """
        if self.file_path is None:
            raise ValueError("File path must be provided")

        # Check if sheet is in exclude list
        if sheet_name in self.exclude_sheets:
            logger.info(f"Skipping excluded sheet '{sheet_name}'")
            return pd.DataFrame()

        # Determine header row to use
        if header is None:
            header = self._get_header_row_for_sheet(sheet_name)

        # Use cached version if available
        cache_key = f"{sheet_name}_{header}_{hash(frozenset(kwargs.items()))}"
        if cache_key in self._dataframes:
            return self._dataframes[cache_key]

        # Combine default kwargs with provided ones
        merged_kwargs = {**self.sheet_load_kwargs, **kwargs}
        # Remove duplicate 'header' if present in both merged_kwargs and as argument
        if 'header' in merged_kwargs and header is not None:
            merged_kwargs.pop('header')
            
        logger.debug(f"Loading sheet '{sheet_name}' from {self.file_path} with header={header}")
        try:
            df = pd.read_excel(
                self.file_path,
                sheet_name=sheet_name,
                header=header,
                **merged_kwargs
            )
            # Cache the result
            self._dataframes[cache_key] = df
            return df
        except Exception as e:
            logger.error(f"Error loading sheet '{sheet_name}': {e}")
            raise

    def load_dataframes(self, file_path: Optional[str | Path] = None) -> Dict[str, pd.DataFrame]:
        """
        Load all sheets or specified sheets from the workbook as DataFrames.

        Parameters
        ----------
        file_path : str | Path, optional
            Path to the Excel workbook. If not provided, uses the instance file_path.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping sheet names to DataFrames.
        """
        if file_path:
            self.file_path = Path(file_path)
            # Clear caches if file path changes
            self._dataframes = {}
            self._sheet_names = None

        if self.file_path is None:
            raise ValueError("File path must be provided")

        result = {}
        
        if self.sheets:
            # Load only the configured sheets
            for logical_name, sheet_config in self.sheets.items():
                sheet_name = sheet_config.get("sheet_name", logical_name)
                if sheet_name in self.exclude_sheets:
                    logger.info(f"Skipping excluded sheet '{sheet_name}'")
                    continue
                
                header = sheet_config.get("header_row", self._get_header_row_for_sheet(sheet_name))
                try:
                    df = self.load_sheet(sheet_name, header)
                    if not df.empty:
                        result[logical_name] = df
                except Exception as e:
                    logger.warning(f"Could not load sheet '{sheet_name}': {e}")
        else:
            # Load all available sheets (except excluded ones)
            try:
                sheet_names = self.get_sheet_list()
                for sheet_name in sheet_names:
                    if sheet_name in self.exclude_sheets:
                        continue
                    try:
                        df = self.load_sheet(sheet_name)
                        if not df.empty:
                            result[sheet_name] = df
                    except Exception as e:
                        logger.warning(f"Could not load sheet '{sheet_name}': {e}")
            except Exception as e:
                logger.error(f"Error loading workbook {self.file_path}: {e}")
                raise
                
        return result

    def load(self, source: Union[str, Path, Iterable[Dict[str, object]]]) -> Iterable[Row]:
        """
        Load documents from source and return as Row objects.

        Parameters
        ----------
        source : Path | str | Iterable[Dict[str, object]]
            Source to load documents from, can be a path or in-memory data

        Returns
        -------
        Iterable[Row]
            Loaded documents as Row objects
        """
        if isinstance(source, (str, Path)):
            self.file_path = Path(source)

        return self._iter()

    def _iter(self) -> Iterator[Row]:
        """
        Iterator that yields each sheet as a Row object.

        Returns
        -------
        Iterator[Row]
            Iterator of Row objects representing Excel sheets
        """
        if self.file_path is None:
            raise ValueError("File path must be provided")

        dataframes = self.load_dataframes()

        for logical_name, df in dataframes.items():
            # Get sheet info from configuration
            sheet_config = self.sheets.get(logical_name, {})
            sheet_name = sheet_config.get("sheet_name", logical_name)
            
            # Create basic metadata
            metadata = {
                "source_type": "excel",
                "sheet_name": sheet_name,
                "logical_name": logical_name,
                "filename": self.file_path.name,
                "file_ext": self.file_path.suffix,
                "row_count": len(df),
                "column_count": len(df.columns),
            }
            
            # Create structured data containing the DataFrame
            structured = {
                "dataframe": df.to_dict(orient="records"),
                "columns": list(df.columns),
                "sheet_config": sheet_config
            }
            
            # Create a minimal text representation
            text = f"Excel sheet: {sheet_name}"
            
            # Generate a unique ID
            row_id = f"excel_{self.file_path.stem}_{sheet_name}_{uuid4().hex[:8]}"
            
            # Yield the Row object with all necessary data
            yield Row(
                text=text,
                metadata=metadata,
                structured=structured,
                assets=[],
                id=row_id
            )

    def _iter_sources(self) -> Iterator[Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]]:
        """
        Iterate through document sources, yielding path, text, metadata, structured data, and assets.

        Yields
        ------
        Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]
            Tuples of (path, text, metadata, structured_data, assets)
        """
        if self.file_path is None:
            raise ValueError("File path must be provided")

        dataframes = self.load_dataframes()
        
        for logical_name, df in dataframes.items():
            sheet_config = self.sheets.get(logical_name, {})
            sheet_name = sheet_config.get("sheet_name", logical_name)
            
            # Create basic metadata
            metadata = {
                "source_type": "excel",
                "sheet_name": sheet_name,
                "logical_name": logical_name,
                "filename": self.file_path.name,
                "file_ext": self.file_path.suffix,
                "row_count": len(df),
                "column_count": len(df.columns),
            }
            
            # Create structured data containing the DataFrame
            structured = {
                "dataframe": df.to_dict(orient="records"),
                "columns": list(df.columns),
                "sheet_config": sheet_config
            }
            
            # Create a minimal text representation (the extractor will enhance this)
            text = f"Excel sheet: {sheet_name}"
            
            # Yield the tuple with all necessary data
            yield (self.file_path, text, metadata, structured, [])

    def get_sheet_list(self) -> List[str]:
        """
        Get a list of sheet names in the workbook.

        Returns
        -------
        List[str]
            List of sheet names.
        """
        if self.file_path is None:
            raise ValueError("File path must be provided")

        # Use cached sheet names if available
        if self._sheet_names is not None:
            return [s for s in self._sheet_names if isinstance(s, str) and s not in self.exclude_sheets]

        try:
            # Use ExcelFile to get sheet names without fully loading the workbook
            with pd.ExcelFile(self.file_path) as xls:
                self._sheet_names = xls.sheet_names
                return [s for s in self._sheet_names if isinstance(s, str) and s not in self.exclude_sheets]
        except Exception as e:
            logger.error(f"Error getting sheet names from {self.file_path}: {e}")
            raise

    def get_sheet_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get basic information about all sheets in the workbook.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping sheet names to information dictionaries.
        """
        if self.file_path is None:
            raise ValueError("File path must be provided")

        sheet_names = self.get_sheet_list()

        result = {}
        for sheet_name in sheet_names:
            if sheet_name in self.exclude_sheets:
                continue

            # Get a sample of the sheet to determine shape and columns
            try:
                header = self._get_header_row_for_sheet(sheet_name)
                sample = pd.read_excel(
                    self.file_path,
                    sheet_name=sheet_name,
                    nrows=5,  # Just get a few rows for info
                    header=header
                )

                result[sheet_name] = {
                    "excluded": sheet_name in self.exclude_sheets,
                    "header_row": header,
                    "columns": list(sample.columns),
                    "sample_rows": len(sample),
                    "estimated_columns": len(sample.columns)
                }
            except Exception as e:
                result[sheet_name] = {
                    "excluded": sheet_name in self.exclude_sheets,
                    "error": str(e)
                }

        return result
