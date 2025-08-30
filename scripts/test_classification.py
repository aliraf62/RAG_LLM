#!/usr/bin/env python
"""
Test script for verifying classification functionality.
"""
import logging
import sys
from pathlib import Path
from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_classification')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline.base import Row
from core.config.settings import settings
from core.services.customer_service import customer_service
from core.utils.component_registry import create_component_instance


def create_test_html():
    """Create a test HTML document."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>How to Create a CSO Optimization Scenario</title>
    </head>
    <body>
        <h1>Step by Step Guide: Creating CSO Optimization Scenarios</h1>
        <p>This guide explains how to set up and configure optimization scenarios in MyCoupa CSO.</p>
        
        <h2>What is a CSO Optimization Scenario?</h2>
        <p>A scenario allows sourcing professionals to create what-if analyses for supplier bid evaluation.</p>
        
        <h2>Step 1: Navigate to the CSO module</h2>
        <p>First, log into your MyCoupa account and navigate to the CSO module from the main dashboard.</p>
        
        <h2>Step 2: Create a New Scenario</h2>
        <p>Click on the "New Scenario" button in the top right corner of the screen.</p>
        
        <h2>Step 3: Configure Scenario Parameters</h2>
        <p>Set up your scenario parameters including constraints, weights, and award strategies.</p>
        
        <h2>Step 4: Run Optimization</h2>
        <p>After configuring all parameters, click the "Run Optimization" button to process your scenario.</p>
    </body>
    </html>
    """


def test_html_classification():
    """Test the classification of HTML content."""
    # Show current settings
    logger.info(f"Customer ID: {customer_service.active_customer}")
    logger.info(f"SMART_CLASSIFY enabled: {settings.SMART_CLASSIFY}")

    # Create sample HTML content
    html_content = create_test_html()

    # Create a sample row - properly initializing with direct attributes
    row = Row(
        text=html_content,
        metadata={
            'title': 'How to Create a CSO Optimization Scenario',
            'url': 'https://mycoupa.com/docs/cso/scenario-creation.html',
        },
        id='test_doc_001'
    )

    # Create HTML cleaner with explicit customer_id
    cleaner = create_component_instance('cleaner', 'html', customer_id='mycoupa')

    # Process the row
    logger.info("Processing row with cleaner...")
    processed_row = cleaner.clean(row)

    # Check if classification was applied
    logger.info("Classification results:")
    logger.info("=" * 80)

    # Print metadata - access as attribute, not dictionary
    if hasattr(processed_row, 'metadata') and processed_row.metadata:
        pprint(processed_row.metadata)
    else:
        logger.warning("No metadata found in processed row!")

    # Check specific classification fields
    classification_fields = ['content_purpose', 'categories', 'tags', 'domains']
    for field in classification_fields:
        if hasattr(processed_row, 'metadata') and field in processed_row.metadata:
            logger.info(f"{field}: {processed_row.metadata.get(field)}")
        else:
            logger.warning(f"Missing expected field: {field}")

    return processed_row


def add_debug_to_base_cleaner():
    """Add debug logging to BaseCleaner to trace classification process."""
    try:
        from core.pipeline.cleaners.base import BaseCleaner

        # Store the original clean method
        original_clean = BaseCleaner.clean

        def debug_clean_wrapper(self, row):
            """Wrapper adding debug logging to classification."""
            logger.info(f"[DEBUG] Classification in clean method with SMART_CLASSIFY={settings.get('SMART_CLASSIFY', False)}")
            logger.info(f"[DEBUG] Using customer_id={self.customer_id}")

            if not settings.get("SMART_CLASSIFY", False):
                logger.info("[DEBUG] Classification might be SKIPPED: SMART_CLASSIFY is disabled")

            logger.info("[DEBUG] Running clean method with potential classification...")
            result = original_clean(self, row)

            # Access the metadata attribute directly, not with get()
            if hasattr(result, 'metadata') and result.metadata:
                logger.info(f"[DEBUG] Clean method complete, metadata keys: {list(result.metadata.keys())}")
            else:
                logger.info("[DEBUG] Clean method complete, no metadata found")

            return result

        # Replace with debug version
        BaseCleaner.clean = debug_clean_wrapper
        logger.info("Added debug logging to BaseCleaner.clean")

    except Exception as e:
        logger.error(f"Failed to add debug logging: {str(e)}")


if __name__ == "__main__":
    # Add debug tracing
    add_debug_to_base_cleaner()

    # Force reload settings and customer
    try:
        # Force customer to mycoupa
        if not customer_service.load_customer('mycoupa', activate=True):
            logger.error("Failed to load customer 'mycoupa'")
            sys.exit(1)

        # Enable SMART_CLASSIFY if not already enabled
        if not settings.get('SMART_CLASSIFY', False):
            logger.info("Forcing SMART_CLASSIFY=True for this run")
            settings.SMART_CLASSIFY = True

        # Run test
        result = test_html_classification()

    except Exception as e:
        logger.exception(f"Error testing classification: {str(e)}")
