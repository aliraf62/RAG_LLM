"""
Simple test to directly load the CSO Excel file
"""
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def main():
    try:
        # Step 1: Load customer settings
        logger.info("Loading customer service module")
        from core.services.customer_service import customer_service
        
        # Step 2: Load config settings
        logger.info("Loading settings module")
        from core.config.settings import settings, apply_customer_settings
        
        # Step 3: Apply customer settings
        customer_id = "coupa"
        logger.info(f"Loading customer {customer_id}")
        if not customer_service.load_customer(customer_id):
            logger.error(f"Failed to load customer {customer_id}")
            return
        
        logger.info(f"Applying customer settings for {customer_id}")
        apply_customer_settings(customer_id)
        
        # Step 4: Check datasets config after loading
        datasets_config = settings.get("datasets", {})
        logger.info(f"Datasets config after loading: {list(datasets_config.keys())}")
        
        if "cso_workflow_guides" in datasets_config:
            cso_config = datasets_config["cso_workflow_guides"]
            logger.info(f"CSO workflow config: {cso_config}")
            logger.info(f"CSO sheets config: {cso_config.get('sheets', {})}")
        else:
            logger.error("'cso_workflow_guides' not found in datasets config")
        
        # Step 5: Try to import the CSOExcelLoader
        logger.info("Trying to import the CSOExcelLoader")
        from customers.coupa.loaders.cso_excel_loader import CSOExcelLoader
        
        # Step 6: Initialize and test
        excel_path = "/Users/ali.rafieefar/Documents/GitHub/ai_qna_assistant/customers/coupa/datasets/in-product-guides-Guide+Export/Workflow Steps.xlsb"
        logger.info(f"Initializing loader with {excel_path}")
        
        loader = CSOExcelLoader(file_path=excel_path, customer_id=customer_id)
        logger.info(f"Loader initialized with sheets config: {loader.sheets}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
