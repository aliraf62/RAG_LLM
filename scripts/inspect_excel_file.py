#!/usr/bin/env python3
"""
Print sheet titles and a sample row from each sheet in an Excel file.
Particularly useful for XLSB files which can be harder to inspect.
"""
import pandas as pd
import sys
from core.config.paths import project_path

def print_excel_info(excel_path):
    """Print sheet names and first row from each sheet in an Excel file."""
    print(f"Analyzing Excel file: {excel_path}")

    try:
        # Load the Excel file
        xls = pd.ExcelFile(excel_path)

        # Get all sheet names
        sheet_names = xls.sheet_names
        print(f"\nFound {len(sheet_names)} sheets:")

        # Process each sheet
        for i, sheet_name in enumerate(sheet_names, start=1):
            print(f"\n{'='*80}\nSheet {i}: {sheet_name}\n{'='*80}")

            # Try different header rows (0, 1, 2, 3) to find column names
            for header_row in range(4):
                try:
                    # Read with this header row
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row, engine='pyxlsb', nrows=2)

                    # Print column names
                    print(f"\nColumn names (with header_row={header_row}):")
                    for j, col in enumerate(df.columns, start=1):
                        print(f"  {j}. {col}")

                    # Print first data row
                    if not df.empty:
                        print(f"\nSample row (first row after header={header_row}):")
                        sample_row = df.iloc[0].to_dict()
                        for col, val in sample_row.items():
                            print(f"  {col}: {val}")
                        break

                except Exception as e:
                    print(f"  Error reading with header_row={header_row}: {e}")

    except Exception as e:
        print(f"Error processing Excel file: {e}")

if __name__ == "__main__":
    # Use provided path or default path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = project_path("customers/coupa/datasets/in-product-guides-Guide+Export/Workflow Steps.xlsb")

    print_excel_info(file_path)
