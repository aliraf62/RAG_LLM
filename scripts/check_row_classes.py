"""
Check for inconsistencies in the Row class definitions
"""
import sys
import inspect
from pathlib import Path

# Add the project root to Python path
root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root))

# Print Python path for debugging
print(f"sys.path: {sys.path}")

try:
    # Try importing different Row classes
    print("\nTrying to import Row from core.utils.models...")
    from core.utils.models import Row as UtilsRow
    print(f"UtilsRow: {UtilsRow}")
    print(f"UtilsRow source: {inspect.getsource(UtilsRow)}")
    
    print("\nTrying to import Row from core.pipeline.base...")
    from core.pipeline.base import Row as PipelineRow
    print(f"PipelineRow: {PipelineRow}")
    print(f"PipelineRow source: {inspect.getsource(PipelineRow)}")
    
    # Check if they're the same
    print(f"\nAre they the same object? {UtilsRow is PipelineRow}")
    
    # Try importing and checking BaseLoader
    print("\nTrying to import BaseLoader...")
    from core.pipeline.loaders.base import BaseLoader
    print(f"BaseLoader: {BaseLoader}")
    print(f"BaseLoader.Row import: {BaseLoader.__module__}.{BaseLoader.__name__}")
    
    # Try importing and checking ExcelLoader
    print("\nTrying to import ExcelLoader...")
    from core.pipeline.loaders.excel_loader import ExcelLoader
    print(f"ExcelLoader: {ExcelLoader}")
    print(f"ExcelLoader bases: {ExcelLoader.__bases__}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
