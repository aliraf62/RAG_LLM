#!/usr/bin/env python3
"""
Script to check if excel and CSO workflow extractor configuration exists in settings.
No customer settings loading, just direct YAML loading.
"""

import sys
import logging
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    project_root = Path(__file__).resolve().parent.parent
    
    # Load the config files directly without using the settings system
    main_config_path = project_root / 'config.yaml'
    customer_config_path = project_root / 'customers/coupa/coupa.yaml'
    
    try:
        with open(main_config_path, 'r') as f:
            main_config = yaml.safe_load(f) or {}
        logger.info(f"Loaded main config: {main_config_path}")
        
        with open(customer_config_path, 'r') as f:
            customer_config = yaml.safe_load(f) or {}
        logger.info(f"Loaded customer config: {customer_config_path}")
        
        # Check for dataset config
        datasets = customer_config.get('datasets', {})
        if datasets:
            logger.info("Customer config has datasets:")
            for dataset_name, dataset in datasets.items():
                logger.info(f"  {dataset_name}:")
                logger.info(f"    file: {dataset.get('file')}")
                logger.info(f"    assets_dir: {dataset.get('assets_dir')}")
        
        # Check for legacy excel config
        excel_config = customer_config.get('excel', {})
        if excel_config:
            logger.info("Customer config has excel config:")
            logger.info(f"  workbook_path: {excel_config.get('workbook_path')}")
            config = excel_config.get('config', {})
            if config:
                logger.info(f"  assets_dir: {config.get('assets_dir')}")
        
        # Check for CSO workflow extractor config
        cso_config = customer_config.get('cso_workflow_extractor', {})
        if cso_config:
            logger.info("Customer config has CSO workflow extractor config:")
            logger.info(f"  assets_dir: {cso_config.get('assets_dir')}")
        
        # If we found at least one path, it's fixable
        if (datasets.get('cso_workflow_guides', {}).get('file') or 
            excel_config.get('workbook_path')):
            logger.info("Found at least one workbook path - config exists")
        else:
            logger.error("No workbook path found in any config!")
            
    except Exception as e:
        logger.error(f"Error loading config files: {e}")

if __name__ == "__main__":
    main()
