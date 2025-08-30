#!/usr/bin/env python3
"""
CSO HTML Exporter Demo
=====================
Simple demo script for exporting workflow guides to HTML files.
Uses default parameters when not specified.
"""
import sys, logging
import argparse
from pathlib import Path

# Set environment variable BEFORE importing any modules that read it
import os

# Parse arguments early to determine caption setting
parser = argparse.ArgumentParser(description="CSO HTML exporter demo")
parser.add_argument("--no-captions", action="store_true",
                    help="Disable image captioning (faster)")
args, _ = parser.parse_known_args()

# DEFAULT IS TO ENABLE CAPTIONS, unless --no-captions is specified
os.environ["SKIP_IMAGE_CAPTIONS"] = "true" if args.no_captions else "false"

# Now import the exporters
from core.pipeline import get_exporter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

def main():
    # Project root (2 levels up from script)
    PROJ_ROOT = Path(__file__).resolve().parents[1]

    # Default paths
    DEFAULT_WB = PROJ_ROOT / "data" / "datasets" / "in-product-guides-Guide+Export" / "Workflow Steps.xlsb"
    DEFAULT_ASSETS = DEFAULT_WB.parent / "WorkflowSteps_unpacked"
    DEFAULT_OUT = PROJ_ROOT / "outputs" / "CSO_workflow_html_exports_demo"
    DEFAULT_LIMIT = 5

    # Parse full arguments
    parser = argparse.ArgumentParser(description="CSO HTML exporter demo")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WB,
                        help=f"Path to Workflow Steps.xlsb")
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS,
                        help=f"Path to assets directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help=f"Output directory")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Guide limit (default: {DEFAULT_LIMIT})")
    parser.add_argument("--no-captions", action="store_true",
                        help="Disable image captioning (faster)")
    args = parser.parse_args()

    # Check if files exist
    if not args.workbook.exists():
        print(f"❌ Error: Workbook not found: {args.workbook}")
        return 1
    if not args.assets.exists():
        print(f"❌ Error: Assets directory not found: {args.assets}")
        return 1
    if not args.no_captions:
        print("🧹 Clearing caption cache to force regeneration...")
        caption_cache_files = list(args.assets.glob("*.caption.json"))
        for cf in caption_cache_files:
            try:
                cf.unlink()
                print(f"   Removed {cf.name}")
            except Exception as e:
                print(f"   Failed to remove {cf.name}: {e}")

    # Show configuration
    print(f"📋 CSO HTML Exporter Demo")
    print(f"   Workbook:        {args.workbook}")
    print(f"   Assets:          {args.assets}")
    print(f"   Output:          {args.output}")
    print(f"   Guide limit:     {args.limit}")
    print(f"   Image captions:  {'Disabled' if os.environ['SKIP_IMAGE_CAPTIONS'] == 'true' else 'Enabled'}")

    try:
        # Get exporter class
        exporter_cls = get_exporter("cso_html")

        # Create exporter instance
        exporter = exporter_cls(
            workbook=args.workbook,
            assets_dir=args.assets,
            out_dir=args.output,
            limit=args.limit
        )

        # Store the limit value for use in the progress tracking
        limit_value = args.limit

        # Add progress tracking hooks (monkey patch logging)
        original_info = logging.Logger.info
        processed_guides = 0

        def info_with_progress(self, msg, *log_args, **kwargs):
            nonlocal processed_guides, limit_value
            if "wrote " in str(msg) and ".html" in str(msg):
                processed_guides += 1
                print(f"✓ [{processed_guides}/{limit_value if limit_value else 'All'}] {msg}",
                      flush=True)
            else:
                original_info(self, msg, *log_args, **kwargs)

        # Apply patch
        logging.Logger.info = info_with_progress

        # Run export
        print(f"🔄 Starting export process...")
        result_dir = exporter.export()

        # Count results
        html_count = sum(1 for _ in result_dir.glob("*.html"))
        img_count = sum(1 for _ in (result_dir / "images").glob("*.*")) if (result_dir / "images").exists() else 0
        file_count = sum(1 for _ in (result_dir / "files").glob("*.*")) if (result_dir / "files").exists() else 0

        # Show summary
        print(f"\n✅ Export complete")
        print(f"   📄 HTML files: {html_count}")
        print(f"   🖼️  Images:     {img_count}")
        print(f"   📎 Other files: {file_count}")
        print(f"   📁 Output path: {result_dir}")
        return 0

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Restore original method
        if 'original_info' in locals():
            logging.Logger.info = original_info


if __name__ == "__main__":
    sys.exit(main())