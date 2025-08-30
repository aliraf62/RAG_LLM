#!/usr/bin/env python3
"""
Minimal script to debug customer settings loading.
"""

import sys
import logging
from pathlib import Path

# Ensure parent directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.config.settings import settings, apply_customer_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load the customer settings
    customer_id = "coupa"
    logger.info(f"Applying customer settings for {customer_id}")
    apply_customer_settings(customer_id)
    
    # Check what's in the settings
    logger.info(f"Available settings keys: {list(settings.keys())}")
    
    # Check for datasets
    if 'datasets' in settings:
        logger.info("'datasets' key is present in settings")
        logger.info(f"Datasets in settings: {list(settings.get('datasets', {}).keys())}")
    else:
        logger.error("'datasets' key is missing from settings")
    
    # Check for legacy excel config
    excel_config = settings.get("excel", {})
    if excel_config:
        logger.info("Excel config is present:")
        logger.info(f"  workbook_path: {excel_config.get('workbook_path')}")
        
        config = excel_config.get("config", {})
        logger.info(f"  assets_dir: {config.get('assets_dir')}")
    else:
        logger.warning("No excel config found")
    
    # Check for CSO workflow extractor config
    cso_config = settings.get("cso_workflow_extractor", {})
    if cso_config:
        logger.info("CSO workflow extractor config is present:")
        logger.info(f"  assets_dir: {cso_config.get('assets_dir')}")
    else:
        logger.warning("No CSO workflow extractor config found")

if __name__ == "__main__":
    main()
