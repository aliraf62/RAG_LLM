#!/usr/bin/env python3
"""
Minimal script to check if Python is running correctly.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Simple Python script running successfully")
    logger.info(f"Python version: {sys.version}")

if __name__ == "__main__":
    main()
