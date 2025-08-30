# tests/test_ingestion.py
# run e.g. in terminal by pytest test_ingestion.py
import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.pipeline.loaders import load_workflow_sections

WORKBOOK = "data/datasets/in-product-guides-Guide+Export/Workflow Steps.xlsb"
ASSETS   = "data/datasets/in-product-guides-Guide+Export/WorkflowSteps_unpacked_transformed"

def test_loader_produces_docs():
    docs = list(load_workflow_sections(WORKBOOK, ASSETS))
    assert len(docs) == 2044                # all sections
    assert all(len(d["text"]) > 50 for d in docs)

def test_asset_paths_exist():
    docs = load_workflow_sections(WORKBOOK, ASSETS)
    png_docs = [p for d in docs for p in d["metadata"]["asset_paths"]]
    assert all(Path(p).exists() for p in png_docs)
