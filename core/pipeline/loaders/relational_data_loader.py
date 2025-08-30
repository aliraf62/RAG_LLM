"""
Relational Data Loader for loading data from relational sources (Excel, SQL, CSV).

This loader handles I/O operations for relational data sources and passes
the raw data to the RelationalDataExtractor for processing. It follows the
gold standard pattern of separating I/O concerns from data processing logic.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import logging
import os

import pandas as pd

from core.pipeline.base import BasePipelineComponent, Row
from core.pipeline.loaders.base import BaseLoader
from core.utils.component_registry import register

logger = logging.getLogger(__name__)


@register("loader", "relational_data")
class RelationalDataLoader(BaseLoader):
    """
    Loader for relational data from various sources (Excel, SQL, CSV).

    This loader handles only I/O operations:
    - Loading data from relational sources (Excel files, database tables)
    - Locating related asset files
    - Passing the raw data to the extractor for processing

    No data processing or relationship handling is done in the loader;
    that responsibility belongs to the RelationalDataExtractor.
    """

    def __init__(
        self,
        source_path: Optional[str] = None,
        assets_dir: Optional[str] = None,
        sheet_names: Optional[List[str]] = None,
        exclude_sheets: Optional[List[str]] = None,
        **config: Any
    ) -> None:
        """
        Initialize the relational data loader.

        Parameters
        ----------
        source_path : str, optional
            Path to the source data (e.g., Excel file path, connection string).
        assets_dir : str, optional
            Directory containing related assets.
        sheet_names : List[str], optional
            Specific sheets to load (Excel only).
        exclude_sheets : List[str], optional
            Sheets to exclude from loading (Excel only).
        config : dict
            Additional configuration parameters.
        """
        super().__init__(**config)
        self.source_path = source_path or config.get("source_path")
        self.assets_dir = assets_dir or config.get("assets_dir")
        self.sheet_names = sheet_names or config.get("sheet_names", [])
        self.exclude_sheets = exclude_sheets or config.get("exclude_sheets", [])

        # Source type detection (Excel, SQL, CSV)
        self.source_type = self._detect_source_type()

    def _detect_source_type(self) -> str:
        """
        Detect the type of data source based on file extension or config.

        Returns
        -------
        str
            Source type: 'excel', 'sql', 'csv', etc.
        """
        if not self.source_path:
            return "unknown"

        # Get explicit source type if provided
        if "source_type" in self.config:
            return self.config["source_type"]

        # Detect from file extension
        path = Path(self.source_path)
        if path.suffix.lower() in [".xlsx", ".xlsb", ".xlsm", ".xls"]:
            return "excel"
        elif path.suffix.lower() in [".csv"]:
            return "csv"
        elif path.suffix.lower() in [".db", ".sqlite", ".sqlite3"]:
            return "sqlite"
        elif "connection_string" in self.config:
            return "sql"
        else:
            return "unknown"

    def _iter_sources(self) -> Iterator[Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]]:
        """
        Iterate through document sources, yielding path, text, metadata, structured data, and assets.

        This implementation delegates to specific loaders based on source_type.

        Yields
        ------
        Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]
            Tuples of (path, text, metadata, structured_data, assets)
        """
        if not self.source_path:
            logger.error("No source path provided")
            return

        path = Path(self.source_path)

        # Handle different source types
        if self.source_type == "excel":
            yield from self._iter_excel_sources(path)
        elif self.source_type == "csv":
            yield from self._iter_csv_sources(path)
        elif self.source_type == "sqlite":
            yield from self._iter_sqlite_sources(path)
        elif self.source_type == "sql":
            yield from self._iter_sql_sources()
        else:
            logger.error(f"Unsupported source type: {self.source_type}")
            return

    def _iter_excel_sources(self, path: Path) -> Iterator[Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]]:
        """
        Iterate through Excel sheets, yielding path, text, metadata, structured data, and assets.

        Parameters
        ----------
        path : Path
            Path to the Excel file.

        Yields
        ------
        Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]
            Tuples of (path, text, metadata, structured_data, assets)
        """
        if not path.exists():
            logger.error(f"Excel file not found: {path}")
            return

        logger.info(f"Loading Excel file: {path}")

        try:
            # Get sheet names if not specified
            excel = pd.ExcelFile(path)
            all_sheets = excel.sheet_names

            # Filter sheet names based on include/exclude lists
            sheets_to_load = set(all_sheets)

            # Apply include filter if specified
            if self.sheet_names:
                sheets_to_load = set(self.sheet_names).intersection(sheets_to_load)

            # Apply exclude filter
            if self.exclude_sheets:
                sheets_to_load = sheets_to_load - set(self.exclude_sheets)

            # Load each sheet
            for sheet_name in sheets_to_load:
                logger.info(f"Loading sheet: {sheet_name}")

                # Read the sheet into a DataFrame
                df = pd.read_excel(path, sheet_name=sheet_name)

                # Skip empty sheets
                if df.empty:
                    logger.warning(f"Sheet '{sheet_name}' is empty, skipping")
                    continue

                # Generate tuple for this sheet
                # The text is empty because the extractor will process DataFrame records
                # Structured data includes the DataFrame records
                # Minimal metadata is included, extractor will add more
                yield (
                    path,  # path
                    "",    # text - empty because the extractor will process DataFrame records
                    {      # metadata
                        "sheet_name": sheet_name,
                        "logical_name": sheet_name,
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "filename": path.name
                    },
                    {      # structured data
                        "dataframe": df.to_dict(orient="records"),
                        "sheet_name": sheet_name
                    },
                    []     # assets - will be linked by extractor
                )

        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return

    def _iter_csv_sources(self, path: Path) -> Iterator[Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]]:
        """
        Iterate through CSV file, yielding path, text, metadata, structured data, and assets.

        Parameters
        ----------
        path : Path
            Path to the CSV file.

        Yields
        ------
        Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]
            Tuples of (path, text, metadata, structured_data, assets)
        """
        if not path.exists():
            logger.error(f"CSV file not found: {path}")
            return

        logger.info(f"Loading CSV file: {path}")

        try:
            # Read the CSV into a DataFrame
            df = pd.read_csv(path, **self.config.get("csv_options", {}))

            # Skip empty files
            if df.empty:
                logger.warning(f"CSV file '{path}' is empty, skipping")
                return

            # Generate tuple for this CSV file
            yield (
                path,  # path
                "",    # text - empty because the extractor will process DataFrame records
                {      # metadata
                    "logical_name": path.stem,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "filename": path.name
                },
                {      # structured data
                    "dataframe": df.to_dict(orient="records"),
                    "sheet_name": path.stem
                },
                []     # assets - will be linked by extractor
            )

        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return

    def _iter_sqlite_sources(self, path: Path) -> Iterator[Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]]:
        """
        Iterate through SQLite tables, yielding path, text, metadata, structured data, and assets.

        Parameters
        ----------
        path : Path
            Path to the SQLite database.

        Yields
        ------
        Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]
            Tuples of (path, text, metadata, structured_data, assets)
        """
        if not path.exists():
            logger.error(f"SQLite database not found: {path}")
            return

        logger.info(f"Loading SQLite database: {path}")

        try:
            import sqlite3

            # Connect to the database
            conn = sqlite3.connect(path)

            # Get table names if not specified
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            all_tables = [row[0] for row in cursor.fetchall()]

            # Filter table names based on include/exclude lists
            tables_to_load = set(all_tables)

            # Apply include filter if specified
            if self.sheet_names:  # Reusing sheet_names for table names
                tables_to_load = set(self.sheet_names).intersection(tables_to_load)

            # Apply exclude filter
            if self.exclude_sheets:  # Reusing exclude_sheets for exclude_tables
                tables_to_load = tables_to_load - set(self.exclude_sheets)

            # Load each table
            for table_name in tables_to_load:
                logger.info(f"Loading table: {table_name}")

                # Read the table into a DataFrame
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql_query(query, conn)

                # Skip empty tables
                if df.empty:
                    logger.warning(f"Table '{table_name}' is empty, skipping")
                    continue

                # Generate tuple for this table
                yield (
                    path,  # path
                    "",    # text - empty because the extractor will process DataFrame records
                    {      # metadata
                        "sheet_name": table_name,  # Using sheet_name for consistency
                        "logical_name": table_name,
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "filename": path.name
                    },
                    {      # structured data
                        "dataframe": df.to_dict(orient="records"),
                        "sheet_name": table_name
                    },
                    []     # assets - will be linked by extractor
                )

            # Close the connection
            conn.close()

        except Exception as e:
            logger.error(f"Error loading SQLite database: {e}")
            return

    def _iter_sql_sources(self) -> Iterator[Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]]:
        """
        Iterate through SQL tables, yielding path, text, metadata, structured data, and assets.

        Yields
        ------
        Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]
            Tuples of (path, text, metadata, structured_data, assets)
        """
        connection_string = self.config.get("connection_string")
        if not connection_string:
            logger.error("No connection string provided for SQL source")
            return

        logger.info("Loading SQL database")

        try:
            import sqlalchemy

            # Create engine
            engine = sqlalchemy.create_engine(connection_string)

            # Get table names if not specified
            inspector = sqlalchemy.inspect(engine)
            all_tables = inspector.get_table_names()

            # Filter table names based on include/exclude lists
            tables_to_load = set(all_tables)

            # Apply include filter if specified
            if self.sheet_names:  # Reusing sheet_names for table names
                tables_to_load = set(self.sheet_names).intersection(tables_to_load)

            # Apply exclude filter
            if self.exclude_sheets:  # Reusing exclude_sheets for exclude_tables
                tables_to_load = tables_to_load - set(self.exclude_sheets)

            # Load each table
            for table_name in tables_to_load:
                logger.info(f"Loading table: {table_name}")

                # Read the table into a DataFrame
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql_query(query, engine)

                # Skip empty tables
                if df.empty:
                    logger.warning(f"Table '{table_name}' is empty, skipping")
                    continue

                # Generate tuple for this table
                # Use Path object with connection string hash for path
                path = Path(f"sql:{hash(connection_string)}/{table_name}")
                yield (
                    path,  # path
                    "",    # text - empty because the extractor will process DataFrame records
                    {      # metadata
                        "connection_string": connection_string,
                        "sheet_name": table_name,  # Using sheet_name for consistency
                        "logical_name": table_name,
                        "row_count": len(df),
                        "column_count": len(df.columns)
                    },
                    {      # structured data
                        "dataframe": df.to_dict(orient="records"),
                        "sheet_name": table_name
                    },
                    []     # assets - will be linked by extractor
                )

        except Exception as e:
            logger.error(f"Error loading SQL database: {e}")
            return

    def load(self) -> Iterable[Row]:
        """
        Load data from the source and yield Row objects with raw data.

        The loader handles only I/O operations without processing relationships.
        The data is passed to the extractor for processing via Row.structured.

        Returns
        -------
        Iterable[Row]
            Row objects containing the raw data in the structured field
        """
        if not self.source_path:
            logger.error("No source path provided")
            return

        # Load data based on source type
        if self.source_type == "excel":
            yield from self._load_excel()
        elif self.source_type == "csv":
            yield from self._load_csv()
        elif self.source_type == "sqlite":
            yield from self._load_sqlite()
        elif self.source_type == "sql":
            yield from self._load_sql()
        else:
            logger.error(f"Unsupported source type: {self.source_type}")
            return

    def _load_excel(self) -> Iterable[Row]:
        """
        Load data from Excel file.

        Returns
        -------
        Iterable[Row]
            Row objects with Excel data in the structured field
        """
        path = Path(self.source_path)
        if not path.exists():
            logger.error(f"Excel file not found: {path}")
            return

        logger.info(f"Loading Excel file: {path}")

        try:
            # Get sheet names if not specified
            excel = pd.ExcelFile(path)
            all_sheets = excel.sheet_names

            # Filter sheet names based on include/exclude lists
            sheets_to_load = set(all_sheets)

            # Apply include filter if specified
            if self.sheet_names:
                sheets_to_load = set(self.sheet_names).intersection(sheets_to_load)

            # Apply exclude filter
            if self.exclude_sheets:
                sheets_to_load = sheets_to_load - set(self.exclude_sheets)

            # Load each sheet
            for sheet_name in sheets_to_load:
                logger.info(f"Loading sheet: {sheet_name}")

                # Read the sheet into a DataFrame
                df = pd.read_excel(path, sheet_name=sheet_name)

                # Skip empty sheets
                if df.empty:
                    logger.warning(f"Sheet '{sheet_name}' is empty, skipping")
                    continue

                # Create a Row object with the DataFrame in the structured field
                # The extractor will process this data based on the schema
                yield Row(
                    text="",  # No text yet, will be generated by extractor
                    metadata={
                        "source": str(path),
                        "sheet_name": sheet_name,
                        "logical_name": sheet_name,  # Can be overridden by schema mapping
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "filename": path.name
                    },
                    structured={
                        "dataframe": df.to_dict(orient="records"),
                        "sheet_name": sheet_name
                    },
                    assets=[],  # Assets will be linked by the extractor
                    id=f"excel:{path.name}:{sheet_name}"
                )

        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return

    def _load_csv(self) -> Iterable[Row]:
        """
        Load data from CSV file.

        Returns
        -------
        Iterable[Row]
            Row objects with CSV data in the structured field
        """
        path = Path(self.source_path)
        if not path.exists():
            logger.error(f"CSV file not found: {path}")
            return

        logger.info(f"Loading CSV file: {path}")

        try:
            # Read the CSV into a DataFrame
            df = pd.read_csv(path, **self.config.get("csv_options", {}))

            # Skip empty files
            if df.empty:
                logger.warning(f"CSV file '{path}' is empty, skipping")
                return

            # Create a Row object with the DataFrame in the structured field
            yield Row(
                text="",  # No text yet, will be generated by extractor
                metadata={
                    "source": str(path),
                    "logical_name": path.stem,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "filename": path.name
                },
                structured={
                    "dataframe": df.to_dict(orient="records"),
                    "sheet_name": path.stem
                },
                assets=[],
                id=f"csv:{path.name}"
            )

        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return

    def _load_sqlite(self) -> Iterable[Row]:
        """
        Load data from SQLite database.

        Returns
        -------
        Iterable[Row]
            Row objects with SQLite table data in the structured field
        """
        path = Path(self.source_path)
        if not path.exists():
            logger.error(f"SQLite database not found: {path}")
            return

        logger.info(f"Loading SQLite database: {path}")

        try:
            import sqlite3

            # Connect to the database
            conn = sqlite3.connect(path)

            # Get table names if not specified
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            all_tables = [row[0] for row in cursor.fetchall()]

            # Filter table names based on include/exclude lists
            tables_to_load = set(all_tables)

            # Apply include filter if specified
            if self.sheet_names:  # Reusing sheet_names for table names
                tables_to_load = set(self.sheet_names).intersection(tables_to_load)

            # Apply exclude filter
            if self.exclude_sheets:  # Reusing exclude_sheets for exclude_tables
                tables_to_load = tables_to_load - set(self.exclude_sheets)

            # Load each table
            for table_name in tables_to_load:
                logger.info(f"Loading table: {table_name}")

                # Read the table into a DataFrame
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql_query(query, conn)

                # Skip empty tables
                if df.empty:
                    logger.warning(f"Table '{table_name}' is empty, skipping")
                    continue

                # Create a Row object with the DataFrame in the structured field
                yield Row(
                    text="",  # No text yet, will be generated by extractor
                    metadata={
                        "source": str(path),
                        "sheet_name": table_name,  # Using sheet_name for consistency
                        "logical_name": table_name,
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "filename": path.name
                    },
                    structured={
                        "dataframe": df.to_dict(orient="records"),
                        "sheet_name": table_name
                    },
                    assets=[],
                    id=f"sqlite:{path.name}:{table_name}"
                )

            # Close the connection
            conn.close()

        except Exception as e:
            logger.error(f"Error loading SQLite database: {e}")
            return

    def _load_sql(self) -> Iterable[Row]:
        """
        Load data from SQL database using connection string.

        Returns
        -------
        Iterable[Row]
            Row objects with SQL table data in the structured field
        """
        connection_string = self.config.get("connection_string")
        if not connection_string:
            logger.error("No connection string provided for SQL source")
            return

        logger.info("Loading SQL database")

        try:
            import sqlalchemy

            # Create engine
            engine = sqlalchemy.create_engine(connection_string)

            # Get table names if not specified
            inspector = sqlalchemy.inspect(engine)
            all_tables = inspector.get_table_names()

            # Filter table names based on include/exclude lists
            tables_to_load = set(all_tables)

            # Apply include filter if specified
            if self.sheet_names:  # Reusing sheet_names for table names
                tables_to_load = set(self.sheet_names).intersection(tables_to_load)

            # Apply exclude filter
            if self.exclude_sheets:  # Reusing exclude_sheets for exclude_tables
                tables_to_load = tables_to_load - set(self.exclude_sheets)

            # Load each table
            for table_name in tables_to_load:
                logger.info(f"Loading table: {table_name}")

                # Read the table into a DataFrame
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql_query(query, engine)

                # Skip empty tables
                if df.empty:
                    logger.warning(f"Table '{table_name}' is empty, skipping")
                    continue

                # Create a Row object with the DataFrame in the structured field
                yield Row(
                    text="",  # No text yet, will be generated by extractor
                    metadata={
                        "source": connection_string,
                        "sheet_name": table_name,  # Using sheet_name for consistency
                        "logical_name": table_name,
                        "row_count": len(df),
                        "column_count": len(df.columns)
                    },
                    structured={
                        "dataframe": df.to_dict(orient="records"),
                        "sheet_name": table_name
                    },
                    assets=[],
                    id=f"sql:{table_name}"
                )

        except Exception as e:
            logger.error(f"Error loading SQL database: {e}")
            return

    def verify_assets_directory(self, path: Optional[str] = None) -> Optional[str]:
        """
        Verify the assets directory exists and is accessible.

        Parameters
        ----------
        path : str, optional
            Assets directory path to verify. If None, uses self.assets_dir.

        Returns
        -------
        Optional[str]
            Absolute path to the assets directory if valid, None otherwise.
        """
        assets_dir = path or self.assets_dir

        if not assets_dir:
            logger.debug("No assets directory specified")
            return None

        assets_path = Path(assets_dir)
        if not assets_path.exists():
            logger.warning(f"Assets directory not found: {assets_path}")
            return None

        if not assets_path.is_dir():
            logger.warning(f"Assets path is not a directory: {assets_path}")
            return None

        return str(assets_path.absolute())

    def list_assets(self, path: Optional[str] = None) -> List[Path]:
        """
        List all files in the assets directory.

        Parameters
        ----------
        path : str, optional
            Assets directory path. If None, uses self.assets_dir.

        Returns
        -------
        List[Path]
            List of file paths in the assets directory.
        """
        assets_dir = path or self.assets_dir

        if not assets_dir:
            return []

        assets_path = Path(assets_dir)
        if not assets_path.exists() or not assets_path.is_dir():
            return []

        # Recursively find all files in the assets directory
        return list(assets_path.glob("**/*"))
