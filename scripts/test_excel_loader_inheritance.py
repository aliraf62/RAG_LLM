"""
Simplified test script to check the core issue with the ExcelLoader class
"""
import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)
print(f"Added {project_root} to Python path")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Check imports
    logger.info("Importing Row from core.utils.models")
    from core.utils.models import Row as ModelsRow
    logger.info(f"ModelsRow: {ModelsRow}")
    
    logger.info("Importing Row from core.pipeline.base")
    from core.pipeline.base import Row as PipelineRow
    logger.info(f"PipelineRow: {PipelineRow}")
    
    logger.info("Importing BaseLoader")
    from core.pipeline.loaders.base import BaseLoader
    logger.info(f"BaseLoader: {BaseLoader}")
    
    logger.info("Importing ExcelLoader")
    from core.pipeline.loaders.excel_loader import ExcelLoader
    logger.info(f"ExcelLoader: {ExcelLoader}")
    
    # Try a simple test class
    class TestLoader(ExcelLoader):
        def __init__(self):
            pass
    
    logger.info("Created TestLoader class successfully")
    
except Exception as e:
    logger.error(f"Error during imports or class creation: {e}")
    import traceback
    traceback.print_exc()
