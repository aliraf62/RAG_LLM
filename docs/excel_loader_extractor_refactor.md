# Excel Loader and Extractor Refactoring

## Overview

This refactoring project separates the responsibilities between Excel loaders and extractors according to the gold standard checklist. The main goal is to make a clean separation of concerns:

1. **Loaders**: Responsible for I/O operations only
   - Reading Excel files from disk
   - Converting sheets to DataFrames
   - Providing basic metadata about the file and sheets
   - NOT applying business logic or data transformation

2. **Extractors**: Responsible for data processing
   - Processing DataFrames provided by loaders
   - Applying business logic and transformations
   - Creating meaningful text representations
   - Extracting metadata and assets
   - Building relationships between entities

## Files Created/Modified

1. `core/pipeline/loaders/excel_loader_fixed.py` - Fixed version of the core Excel loader
2. `core/pipeline/extractors/excel_extractor_fixed.py` - Fixed version of the core Excel extractor
3. `customers/coupa/loaders/cso_excel_loader_new.py` - Fixed CSO-specific loader
4. `customers/coupa/extractors/cso_excel_extractor.py` - New CSO-specific extractor
5. `scripts/test_cso_excel_separation.py` - Test script to verify proper separation

## Key Improvements

- Proper separation of concerns following the gold standard checklist
- Loaders now focus solely on loading data, not processing it
- Extractors handle all data transformation and relationship building
- CSO-specific logic moved to the CSO extractor, not the loader
- Clean interfaces between components
- Better maintainability and extensibility
- Robust string handling for all data types in text formatting
- Consistent error handling throughout the components

## Implementation Details

### Core Excel Loader
- Reads Excel files and converts sheets to DataFrames
- Creates minimal Row objects with DataFrames in the structured field
- Provides configuration options for sheet loading
- No business logic or data transformation

### Core Excel Extractor
- Processes DataFrames provided by the loader
- Creates meaningful text representations
- Extracts metadata and assets
- Provides methods for row formatting and asset processing

### CSO Excel Loader
- Extends core Excel loader with CSO-specific configuration
- Handles loading of specific sheets needed for CSO data
- Registers metadata schemas if present in configuration
- Adds basic CSO metadata to rows

### CSO Excel Extractor
- Implements CSO-specific business logic
- Builds relationships between guides, steps, sections, and docs
- Creates hierarchical text representations
- Extracts assets from document fields

## Testing

Use the test script to verify the separation:

```bash
python scripts/test_cso_excel_separation.py [path/to/excel/file]
```

This will demonstrate the clean handoff between loader and extractor.
