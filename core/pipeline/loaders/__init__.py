"""
Core pipeline loaders initialization.

This file ensures all loader components are properly registered with the component registry.
"""

from core.pipeline.loaders.base import BaseLoader

# Import all loader implementations to ensure they're registered
from core.pipeline.loaders.excel_loader import ExcelLoader
from core.pipeline.loaders.html_fs_loader import HTMLFSLoader
from core.pipeline.loaders.parquet_document_loader import ParquetDocumentLoader
from core.pipeline.loaders.relational_data_loader import RelationalDataLoader

# Import our new relational data loader
from core.pipeline.loaders.relational_data_loader import RelationalDataLoader

# Additional loaders can be imported here
