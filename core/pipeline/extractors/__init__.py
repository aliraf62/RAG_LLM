"""
Core pipeline extractors initialization.

This file ensures all extractor components are properly registered with the component registry.
"""

from core.pipeline.extractors.base import BaseExtractor

# Import our new relational data extractor - this is the component we want to register
from core.pipeline.extractors.relational_data_extractor import RelationalDataExtractor

# No need to explicitly import other extractors that are already imported elsewhere
# in the application startup process
