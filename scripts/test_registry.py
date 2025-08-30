"""
Simple script to test customer module loading and check registry
"""
import logging
from core.services.customer_service import customer_service
from core.config.settings import apply_customer_settings
from core.utils.component_registry import _REGISTRY
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def main():
    # Load customer modules
    customer_id = "coupa"
    logger.info(f"Loading customer {customer_id}")
    customer_service.load_customer(customer_id)
    
    # Apply customer settings
    logger.info(f"Applying customer settings for {customer_id}")
    apply_customer_settings(customer_id)
    
    # Check if the loader is registered
    logger.info("Checking registered components")
    for category, components in _REGISTRY.items():
        logger.info(f"Category: {category}")
        for name, factory in components.items():
            logger.info(f"  - {name}: {factory.__module__}")
    
    # Check for the CSO excel loader specifically
    if "loader" in _REGISTRY and "cso_excel" in _REGISTRY["loader"]:
        logger.info("CSO Excel loader is registered!")
        factory = _REGISTRY["loader"]["cso_excel"]
        logger.info(f"Factory: {factory.__module__}.{factory.__name__}")
    else:
        logger.error("CSO Excel loader is NOT registered")
        if "loader" in _REGISTRY:
            logger.info(f"Available loaders: {list(_REGISTRY['loader'].keys())}")
    
    # Try importing the customer module directly
    try:
        logger.info("Trying to import CSO Excel loader directly")
        from customers.coupa.loaders.cso_excel_loader import CSOExcelLoader
        logger.info(f"Successfully imported CSOExcelLoader from {CSOExcelLoader.__module__}")
    except ImportError as e:
        logger.error(f"Failed to import CSOExcelLoader: {e}")
    
    # Check paths
    import sys
    logger.info(f"Python path: {sys.path}")

if __name__ == "__main__":
    main()
