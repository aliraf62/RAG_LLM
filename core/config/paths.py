"""Path utilities for project-wide path resolution.

Provides utilities for finding project root and constructing absolute paths
relative to the project root, regardless of where scripts are executed from.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from core.config.base_paths import (
    PROJECT_ROOT,
    CUSTOMERS_DIR,
    SHARED_DIR,
    find_project_root,
    project_path,
    validate_safe_path,
    validate_customer_id,
    PathValidationError,
    sanitize_path_component
)

logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

def get_customer_root(customer_id: str) -> Path:
    """Get the root directory for a customer.

    Args:
        customer_id: Customer identifier

    Returns:
        Path to customer root directory

    Raises:
        PathValidationError: If customer_id is invalid or path would be unsafe
    """
    validated_id = validate_customer_id(customer_id)
    customer_dir = CUSTOMERS_DIR / validated_id  # Renamed from customer_path to avoid shadowing
    return validate_safe_path(customer_dir, CUSTOMERS_DIR)

# Alias for backward compatibility (legacy code expects customer_path)
customer_path = get_customer_root

def get_customer_outputs_dir(customer_id: str, dataset_id: Optional[str] = None) -> Path:
    """Get customer-specific outputs directory.

    Args:
        customer_id: Customer identifier
        dataset_id: Optional dataset identifier

    Returns:
        Path to customer's outputs directory

    Raises:
        PathValidationError: If paths would be unsafe
    """
    base = get_customer_root(customer_id) / "outputs"
    if dataset_id:
        safe_dataset = sanitize_path_component(dataset_id)
        path = enforce_path_safety(base / safe_dataset, CUSTOMERS_DIR)
    else:
        path = validate_safe_path(base, CUSTOMERS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_customer_vector_store_dir(customer_id: str, dataset_id: Optional[str] = None, backend: str = "faiss") -> Path:
    """Get customer-specific vector store directory.

    Args:
        customer_id: Customer identifier
        dataset_id: Optional dataset identifier
        backend: Vector store backend type

    Returns:
        Path to customer's vector store directory

    Raises:
        PathValidationError: If paths would be unsafe
    """
    validated_id = validate_customer_id(customer_id)
    safe_backend = sanitize_path_component(backend)
    base = get_customer_root(validated_id) / "vector_store"
    if dataset_id:
        safe_dataset = sanitize_path_component(dataset_id)
        path = base / f"{safe_dataset}_{safe_backend}"
    else:
        path = base / f"all_datasets_{safe_backend}"
    safe_path = enforce_path_safety(path, CUSTOMERS_DIR)
    safe_path.mkdir(parents=True, exist_ok=True)
    return safe_path

def get_customer_assets_dir(customer_id: str, dataset_id: Optional[str] = None) -> Path:
    """Get customer-specific assets directory.

    Args:
        customer_id: Customer identifier
        dataset_id: Optional dataset identifier

    Returns:
        Path to customer's assets directory

    Raises:
        PathValidationError: If paths would be unsafe
    """
    base = get_customer_outputs_dir(customer_id)
    if dataset_id:
        safe_dataset = sanitize_path_component(dataset_id)
        path = enforce_path_safety(base / safe_dataset / "assets", CUSTOMERS_DIR)
    else:
        path = validate_safe_path(base / "assets", CUSTOMERS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_customer_datasets_dir(customer_id: str) -> Path:
    """Get customer-specific datasets directory.

    Args:
        customer_id: Customer identifier

    Returns:
        Path to customer's datasets directory

    Raises:
        PathValidationError: If paths would be unsafe
    """
    validated_id = validate_customer_id(customer_id)
    path = enforce_path_safety(get_customer_root(validated_id) / "datasets", CUSTOMERS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_customer_config_dir(customer_id: str) -> Path:
    """Get customer-specific configuration directory.

    Args:
        customer_id: Customer identifier

    Returns:
        Path to customer's configuration directory

    Raises:
        PathValidationError: If paths would be unsafe
    """
    validated_id = validate_customer_id(customer_id)
    return enforce_path_safety(get_customer_root(validated_id) / "config", CUSTOMERS_DIR)

def ensure_customer_dirs(customer_id: str) -> None:
    """Ensure all required customer directories exist safely.

    Args:
        customer_id: Customer identifier

    Raises:
        PathValidationError: If customer_id is invalid or paths would be unsafe
    """
    validated_id = validate_customer_id(customer_id)
    root = get_customer_root(validated_id)
    for subdir in ["outputs", "vector_store", "assets", "datasets", "config"]:
        path = enforce_path_safety(root / subdir, CUSTOMERS_DIR)
        path.mkdir(parents=True, exist_ok=True)
