"""
Configuration handling for the application.

This package contains modules for loading, managing, and accessing
configuration data across the application following gold standards:
- Settings hierarchy (defaults → root → customer)
- Path isolation per customer
- Component configuration via registry
"""

from core.config.settings import settings, load_settings, apply_customer_settings
from core.config.paths import (
    get_customer_root,
    get_customer_outputs_dir,
    get_customer_vector_store_dir,
    get_customer_assets_dir,
    project_path,
    find_project_root,
    customer_path  # Maintain backward compatibility
)

__all__ = [
    'settings',
    'load_settings',
    'apply_customer_settings',
    'project_path',
    'find_project_root',
    'get_customer_root',
    'customer_path',  # For backward compatibility
    'get_customer_outputs_dir',
    'get_customer_vector_store_dir',
    'get_customer_assets_dir',
]
