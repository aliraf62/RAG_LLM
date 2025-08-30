#!/usr/bin/env python
"""
Test script for the RelationalDataLoader and RelationalDataExtractor with CSO workflow data.

This script tests the new relational data components with the CSO workflow Excel file.
It loads a small subset of data and verifies that the complex relationships are properly extracted.
"""

import logging
import os
from pathlib import Path
import sys
import json
import pprint

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config.settings import settings
from core.services.customer_service import customer_service
from core.utils.component_registry import create_component_instance

# Import loaders and extractors packages to ensure our components are registered
import core.pipeline.loaders
import core.pipeline.extractors

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Customer ID to use
CUSTOMER_ID = "mycoupa"

def test_relational_data_pipeline():
    """Test the relational data loader and extractor with CSO workflow data."""
    # Initialize the customer configuration
    logger.info(f"Loading customer: {CUSTOMER_ID}")

    # Use load_customer method instead of setting active_customer property directly
    if not customer_service.load_customer(CUSTOMER_ID, activate=True):
        logger.error(f"Failed to load customer: {CUSTOMER_ID}")
        return

    customer_config = customer_service.get_customer_config(CUSTOMER_ID)

    # Get the dataset configuration from datasets section
    if "datasets" not in customer_config:
        logger.error("No datasets section found in customer configuration")
        return

    dataset_config = customer_config["datasets"].get("cso_workflow_relational")
    if not dataset_config:
        logger.error("Dataset 'cso_workflow_relational' not found in customer configuration")
        return

    # Log configuration details
    logger.info(f"Testing with customer ID: {CUSTOMER_ID}")
    logger.info(f"Dataset: {dataset_config.get('description', 'No description')}")
    logger.info(f"Excel file: {dataset_config.get('file')}")
    logger.info(f"Schema config: {dataset_config.get('schema_config')}")
    logger.info(f"Assets directory: {dataset_config.get('assets_dir')}")

    # Create the loader using the component registry
    loader_provider = dataset_config.get("loader")
    loader_config = dataset_config.get("loader_config", {})
    loader_config.update({
        "customer_id": CUSTOMER_ID,
        "source_path": dataset_config.get("file"),
        "assets_dir": dataset_config.get("assets_dir"),
    })

    logger.info(f"Creating loader with provider: {loader_provider}")
    loader = create_component_instance("loader", loader_provider, **loader_config)

    # Load the data
    logger.info("Loading data...")
    input_rows = list(loader.load())
    logger.info(f"Loaded {len(input_rows)} sheets/tables")

    # Display sheet information
    for row in input_rows:
        sheet_name = row.metadata.get("sheet_name")
        if "dataframe" in row.structured:
            records = row.structured["dataframe"]
            logger.info(f"Sheet '{sheet_name}': {len(records)} rows")

    # Create the extractor using the component registry
    extractor_provider = dataset_config.get("extractor")
    extractor_config = dataset_config.get("extractor_config", {})
    extractor_config.update({
        "customer_id": CUSTOMER_ID,
        "schema_config": dataset_config.get("schema_config"),
        "rows": input_rows,
        # Explicitly set export_limit to 3, overriding any value from the configuration
        "export_limit": 3,
    })

    logger.info(f"Creating extractor with provider: {extractor_provider}")
    extractor = create_component_instance("extractor", extractor_provider, **extractor_config)

    # Extract the data
    logger.info("Extracting relationships...")
    output_rows = list(extractor.extract_rows())
    logger.info(f"Extracted {len(output_rows)} rows")

    # Display extracted data summary by entity type
    entity_counts = {}
    entity_rows = {}

    # Group rows by entity type for easy reference
    for row in output_rows:
        entity_name = row.metadata.get("entity_name", "unknown")
        entity_counts[entity_name] = entity_counts.get(entity_name, 0) + 1
        if entity_name not in entity_rows:
            entity_rows[entity_name] = []
        entity_rows[entity_name].append(row)

    # Show entity count summary
    logger.info("\n=== EXTRACTION SUMMARY ===")
    for entity, count in entity_counts.items():
        logger.info(f"{entity}: {count} rows")

    # Find guide entities to analyze in more detail
    if 'guide' in entity_rows:
        logger.info("\n=== DETAILED GUIDE ANALYSIS ===")
        for idx, guide_row in enumerate(entity_rows['guide']):
            # Skip header rows
            if guide_row.metadata.get("entity_id") in ["nan", "Id"]:
                continue

            guide_id = guide_row.metadata.get("entity_id")
            guide_name = guide_row.metadata.get("Name", "Unnamed Guide")

            logger.info(f"\n[GUIDE {idx+1}] {guide_name} (ID: {guide_id})")
            logger.info(f"Text preview: {guide_row.text[:150]}...")

            # Show metadata
            logger.info("Metadata:")
            for key, value in guide_row.metadata.items():
                if key != "metadata":  # Skip nested metadata
                    logger.info(f"  {key}: {value}")

            # Show relationships from structured data
            if "relationships" in guide_row.structured:
                logger.info("\nRelationships:")
                relationships = guide_row.structured["relationships"]

                # Check for steps
                if "guideStep" in relationships:
                    steps = relationships["guideStep"].get("data", [])
                    logger.info(f"  Steps: {len(steps)} related steps")

                    for step_idx, step in enumerate(steps[:2]):  # Show first 2 steps
                        step_id = step.get("id")
                        step_data = step.get("data", {})
                        step_title = step_data.get("title", "Unnamed Step")
                        logger.info(f"    Step {step_idx+1}: {step_title} (ID: {step_id})")

                        # Search for sections related to this step
                        step_sections = []
                        for section_row in output_rows:
                            if (section_row.metadata.get("entity_name") == "guideStepSection" and
                                section_row.metadata.get("StepId") == step_id):
                                step_sections.append(section_row)

                        if step_sections:
                            logger.info(f"      Sections: {len(step_sections)} found")
                            for sec_idx, section in enumerate(step_sections[:2]):  # Show first 2 sections
                                section_type = section.metadata.get("type")
                                section_content = _extract_section_content(section)
                                logger.info(f"        Section {sec_idx+1}: Type: {section_type}, Content: {section_content}")

                                # Show assets if any
                                if section.assets:
                                    logger.info(f"          Assets: {section.assets}")

                    if len(steps) > 2:
                        logger.info(f"    ... and {len(steps) - 2} more steps")

                # Check for folder relations
                # This shows if we're correctly identifying which folders a guide belongs to
                categories = guide_row.metadata.get("categories", [])
                if categories:
                    logger.info(f"  Categories: {', '.join(categories)}")

            # Show assets directly linked to guide
            if guide_row.assets:
                logger.info(f"Assets: {guide_row.assets}")

    return output_rows

def _extract_section_content(section_row):
    """Extract readable content from a section row"""
    content = ""
    if "content" in section_row.metadata:
        try:
            content_json = json.loads(section_row.metadata["content"])
            if "text" in content_json:
                return content_json["text"][:50] + "..." if len(content_json["text"]) > 50 else content_json["text"]
            elif "image" in content_json:
                return f"Image: {content_json['image'].get('name', 'Unnamed')}"
            elif "childId" in content_json:
                return f"Child step reference: {content_json['childId']}"
            elif "guide" in content_json:
                return f"Subguide reference: {content_json['guide'].get('name', 'Unnamed')}"
            else:
                return f"Content: {str(content_json)[:50]}..."
        except:
            return f"Raw content: {section_row.metadata['content'][:50]}..."
    return "No content"

if __name__ == "__main__":
    logger.info("=== TESTING RELATIONAL DATA PIPELINE ===")
    results = test_relational_data_pipeline()
    logger.info("=== TEST COMPLETE ===")

