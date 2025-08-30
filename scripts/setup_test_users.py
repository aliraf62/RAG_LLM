#!/usr/bin/env python
"""
scripts/setup_test_users.py
===========================

Script to create test users for development and testing.

This script creates user accounts and credentials for testing
the authentication system without setting up a full database.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our auth modules
from core.auth.user import create_test_users, UserStore

def setup_default_users():
    """Create default test users for development."""
    logger.info("Setting up default test users...")

    # Create test users for the coupa customer
    create_test_users("coupa", [
        {
            "username": "ali.rafieefar",
            "email": "ali.rafieefar@coupa.com",
            "password": "password1",
            "full_name": "Ali Rafieefar",
            "roles": ["admin", "user"]
        },
        {
            "username": "test.user",
            "email": "test.user@coupa.com",
            "password": "password123",
            "full_name": "Test User",
            "roles": ["user"]
        }
    ])

    logger.info("Default test users created successfully.")

def setup_customer_users(customer_id, users):
    """Create test users for a specific customer."""
    logger.info(f"Setting up test users for customer: {customer_id}")
    create_test_users(customer_id, users)
    logger.info(f"Test users for {customer_id} created successfully.")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Set up test users for development and testing")
    parser.add_argument("--customer", help="Customer ID to create users for")
    parser.add_argument("--users", help="JSON file containing user definitions")
    args = parser.parse_args()

    if args.customer and args.users:
        import json
        with open(args.users, 'r') as f:
            users = json.load(f)
        setup_customer_users(args.customer, users)
    else:
        setup_default_users()

if __name__ == "__main__":
    main()
