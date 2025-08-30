"""Test if the ExcelLoader can be properly imported and subclassed"""
import sys
from pathlib import Path

# Add the project root to Python path
root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root))

# Try importing ExcelLoader
try:
    from core.pipeline.loaders.excel_loader import ExcelLoader
    print(f"Successfully imported ExcelLoader from core.pipeline.loaders.excel_loader")
    
    print(f"ExcelLoader.__bases__ = {ExcelLoader.__bases__}")
    
    # Try creating a basic subclass
    class SimpleExcelLoader(ExcelLoader):
        def __init__(self):
            # Skip calling super().__init__() to avoid complex initialization
            pass
    
    print(f"Successfully created SimpleExcelLoader class")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
