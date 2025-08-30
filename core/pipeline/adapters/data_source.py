# pipeline/adapters/data_source.py
# keeping it for now. should implement to have a pluggable data source system.
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

class DataSourceAdapter(ABC):
    """Abstract adapter for different data sources."""

    @abstractmethod
    def load_data(self) -> Any:
        """Load data from source into standardized format."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Get a name identifying this data source."""
        pass


class ExcelAdapter(DataSourceAdapter):
    """Handles Excel files (XLSX, XLSB, etc.)"""

    def __init__(self, file_path: Path, sheet_names: Optional[List[str]] = None,
                 engine: Optional[str] = None, header_rows: Dict[str, int] = None):
        self.file_path = Path(file_path)
        self.sheet_names = sheet_names
        self.engine = engine or ('pyxlsb' if file_path.suffix.lower() == '.xlsb' else 'openpyxl')
        self.header_rows = header_rows or {}

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load Excel sheets into dataframes.

        Returns:
            Dict mapping sheet names to dataframes
        """
        data = {}

        # If specific sheets requested, load only those
        if self.sheet_names:
            for sheet_name in self.sheet_names:
                header = self.header_rows.get(sheet_name, 0)
                data[sheet_name] = pd.read_excel(
                    self.file_path,
                    sheet_name=sheet_name,
                    engine=self.engine,
                    header=header
                )
        else:
            # Load all sheets
            sheets = pd.read_excel(
                self.file_path,
                sheet_name=None,  # None means all sheets
                engine=self.engine
            )
            data.update(sheets)

        return data

    @property
    def source_name(self) -> str:
        return f"excel:{self.file_path.stem}"


class ParquetAdapter(DataSourceAdapter):
    """Handles Parquet files."""

    def __init__(self, file_path: Path, columns: Optional[List[str]] = None):
        self.file_path = Path(file_path)
        self.columns = columns

    def load_data(self) -> pd.DataFrame:
        """Load Parquet file into dataframe.

        Returns:
            DataFrame with the parquet data
        """
        if self.columns:
            return pd.read_parquet(self.file_path, columns=self.columns)
        return pd.read_parquet(self.file_path)

    @property
    def source_name(self) -> str:
        return f"parquet:{self.file_path.stem}"


class TextFileAdapter(DataSourceAdapter):
    """Handles text files."""

    def __init__(self, file_path: Path, encoding: str = 'utf-8'):
        self.file_path = Path(file_path)
        self.encoding = encoding

    def load_data(self) -> str:
        """Load text file content.

        Returns:
            String content of the file
        """
        return self.file_path.read_text(encoding=self.encoding)

    @property
    def source_name(self) -> str:
        return f"text:{self.file_path.stem}"


class DirectoryAdapter(DataSourceAdapter):
    """Handles directories of files."""

    def __init__(self, dir_path: Path, pattern: str = "*", recursive: bool = False):
        self.dir_path = Path(dir_path)
        self.pattern = pattern
        self.recursive = recursive

    def load_data(self) -> Dict[Path, bytes]:
        """Load files from directory.

        Returns:
            Dict mapping file paths to file contents
        """
        glob_pattern = "**/" + self.pattern if self.recursive else self.pattern
        files = self.dir_path.glob(glob_pattern)

        result = {}
        for file_path in files:
            if file_path.is_file():
                result[file_path] = file_path.read_bytes()

        return result

    @property
    def source_name(self) -> str:
        return f"dir:{self.dir_path.name}"