"""
Test script to run the Excel loader in isolation and inspect its output.
"""
import sys
import argparse
from rich.console import Console
from rich.logging import RichHandler
import logging
from core.config.settings import settings, apply_customer_settings
from core.services.customer_service import customer_service
from pathlib import Path

# Ensure customer modules are loaded
customer_service.load_customer('coupa')
apply_customer_settings('coupa')

console = Console()

# Path to a sample Excel file (adjust as needed)
EXCEL_PATH = "/Users/ali.rafieefar/Documents/GitHub/ai_qna_assistant/customers/coupa/datasets/in-product-guides-Guide+Export/Workflow Steps.xlsb"


def main():
    parser = argparse.ArgumentParser(description="Test loader in isolation with diagnostics.")
    parser.add_argument("--customer", default="coupa", help="Customer name")
    parser.add_argument(
        "--dataset",
        default=EXCEL_PATH,
        help="Dataset file (default: docs/batch_answers.xlsx)"
    )
    parser.add_argument("--loader", default="excel", help="Loader type (excel, html, parquet, etc.)")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of docs (default: 5)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[RichHandler(console=console, show_time=True, show_level=True, show_path=False)])
    logger = logging.getLogger("loader_test")

    console.print(f"[yellow]Loader type: {args.loader} | Dataset: {args.dataset} | Customer: {args.customer}[/yellow]")
    if args.loader == "excel":
        # Try to use CSOExcelLoader if available for customer-specific config
        try:
            from customers.coupa.loaders.cso_excel_loader import CSOExcelLoader
            loader = CSOExcelLoader(file_path=args.dataset, customer_id=args.customer)
            console.print("[green]Using CSOExcelLoader for Coupa customer.[/green]")
        except ImportError:
            from core.pipeline.loaders.excel_loader import ExcelLoader
            loader = ExcelLoader(file_path=args.dataset, customer_id=args.customer)
            console.print("[yellow]Falling back to generic ExcelLoader.[/yellow]")
        loaded_data = loader.load(args.dataset)
        if loaded_data is not None and not isinstance(loaded_data, dict):
            loaded_data = list(loaded_data)
            console.print(f"[red]DEBUG: loaded_data type: {type(loaded_data)}, first element: {loaded_data[0] if loaded_data else 'EMPTY'}[/red]")
            if loaded_data and isinstance(loaded_data[0], tuple) and len(loaded_data[0]) == 2:
                loaded_data = dict(loaded_data)
            elif loaded_data and all(hasattr(df, 'columns') for df in loaded_data):
                loaded_data = {f"Sheet{i+1}": df for i, df in enumerate(loaded_data)}
            elif loaded_data and all(type(row).__name__ == "Row" for row in loaded_data):
                pass
            elif not loaded_data:
                loaded_data = {}
            else:
                raise ValueError(f"Loaded data is not a dict, list of pairs, list of DataFrames, or list of Row objects. Type: {type(loaded_data)}")
        logger.info(f"Loaded {len(loaded_data) if loaded_data else 0} sheets/rows from dataset {args.dataset}")
        if loaded_data:
            if isinstance(loaded_data, dict):
                for sheet_name, df in loaded_data.items():
                    logger.info(f"  Sheet '{sheet_name}': {len(df)} rows, columns: {list(df.columns)}")
                    if sheet_name.lower() == "guidedoc":
                        console.print(f"[bold yellow]DEBUG: guideDoc columns: {list(df.columns)}[/bold yellow]")
            elif isinstance(loaded_data, list):
                logger.info(f"Loaded_data is a list with {len(loaded_data)} elements (not a dict of sheets).")
                for i, row in enumerate(loaded_data[:args.limit]):
                    console.print(f"Row {i}:")
                    console.print(f"  text: {repr(getattr(row, 'text', None))}")
                    console.print(f"  metadata: {getattr(row, 'metadata', None)}")
                    console.print(f"  __dict__: {row.__dict__ if hasattr(row, '__dict__') else str(row)}")
            else:
                logger.info(f"Loaded_data type: {type(loaded_data)}")
    else:
        from core.pipeline.loaders import create_loader
        loader = create_loader(args.loader, root_dir=args.dataset)
        loaded_data = list(loader) if hasattr(loader, '__iter__') else list(loader.load(args.dataset))
        logger.info(f"Loaded {len(loaded_data)} items from dataset {args.dataset} using loader '{args.loader}'")
        for i, row in enumerate(loaded_data[:args.limit]):
            console.print(f"Row {i}:")
            console.print(f"  text: {repr(getattr(row, 'text', None))}")
            console.print(f"  metadata: {getattr(row, 'metadata', None)}")
            console.print(f"  __dict__: {row.__dict__ if hasattr(row, '__dict__') else str(row)}")

if __name__ == "__main__":
    main()
