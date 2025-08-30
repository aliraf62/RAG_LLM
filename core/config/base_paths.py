"""Base path utilities that don't depend on settings.

Contains only essential paths and functions that don't create circular dependencies.
All settings-dependent path operations are in paths.py.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

class PathValidationError(Exception):
    """Raised when path validation fails."""
    pass

__all__ = [
    "PROJECT_ROOT",
    "CUSTOMERS_DIR",
    "SHARED_DIR",
    "find_project_root",
    "project_path",
    "validate_safe_path",
    "enforce_path_safety",
    "sanitize_path_component",
    "validate_customer_id",
    "PathValidationError"
]

# Core directories that don't depend on settings
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUSTOMERS_DIR = PROJECT_ROOT / "customers"  # Base dir for all customer data
SHARED_DIR = PROJECT_ROOT / "shared"     # Shared resources

# Path security constants
SAFE_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')
SAFE_CUSTOMER_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$')
MAX_PATH_LENGTH = 255
RESTRICTED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
    'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3',
    'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
}

def validate_customer_id(customer_id: str) -> str:
    """Validate a customer ID is safe.

    Args:
        customer_id: Customer identifier to validate

    Returns:
        Validated customer ID

    Raises:
        PathValidationError: If customer ID is invalid
    """
    if not customer_id:
        raise PathValidationError("Customer ID cannot be empty")

    if not SAFE_CUSTOMER_PATTERN.match(customer_id):
        raise PathValidationError(
            f"Customer ID '{customer_id}' contains invalid characters. "
            "Must contain only letters, numbers, underscores and hyphens."
        )

    if len(customer_id) > 64:  # Reasonable limit for customer IDs
        raise PathValidationError(f"Customer ID '{customer_id}' is too long")

    return customer_id

def sanitize_path_component(component: str) -> str:
    """Sanitize a path component to make it safe."""
    # Skip path separator components
    if component in ['/', os.sep]:
        return component

    # Skip drive letters on Windows (e.g., "C:")
    if len(component) == 2 and component[1] == ':':
        return component

    # Remove any characters not matching safe pattern
    sanitized = ''.join(c for c in component if SAFE_PATH_PATTERN.match(c))

    # Truncate to maximum length
    sanitized = sanitized[:MAX_PATH_LENGTH]

    # Ensure we still have a valid string
    if not sanitized or not SAFE_PATH_PATTERN.match(sanitized):
        raise PathValidationError(f"Path component '{component}' cannot be made safe")

    return sanitized

def validate_safe_path(
    path: Union[str, Path],
    root: Optional[Path] = None,
    allow_patterns: Optional[List[str]] = None
) -> Path:
    """Validate a path is safe and normalize it."""
    try:
        normalized = Path(path).resolve()
    except Exception as e:
        raise PathValidationError(f"Invalid path '{path}': {e}")

    # Skip validation for root path components
    path_parts = list(normalized.parts)
    if len(path_parts) > 0 and path_parts[0] == '/':
        path_parts = path_parts[1:]

    # On Windows, skip drive letter
    if len(path_parts) > 0 and ':' in path_parts[0]:
        path_parts = path_parts[1:]

    # Check remaining components
    for part in path_parts:
        if part in RESTRICTED_NAMES:
            raise PathValidationError(f"Path contains restricted name: {part}")

        # Skip path separators
        if part in ['/', os.sep]:
            continue

        # Check if part matches any allowed patterns
        if allow_patterns:
            if not any(re.match(pattern, part) for pattern in allow_patterns):
                raise PathValidationError(f"Path component '{part}' doesn't match allowed patterns")
        # Otherwise use default safe pattern
        elif not SAFE_PATH_PATTERN.match(part):
            raise PathValidationError(f"Path component '{part}' contains invalid characters")

    # Verify path length
    if len(str(normalized)) > MAX_PATH_LENGTH:
        raise PathValidationError(f"Path exceeds maximum length of {MAX_PATH_LENGTH}")

    # If root provided, ensure path is under it
    if root:
        try:
            normalized.relative_to(root)
        except ValueError:
            raise PathValidationError(f"Path '{path}' is not under root '{root}'")

    return normalized

def enforce_path_safety(path: Union[str, Path], root: Optional[Path] = None) -> Path:
    """Enforce path safety by sanitizing and validating."""
    # Convert to Path and get parts
    path_obj = Path(path)

    # Handle absolute paths
    if path_obj.is_absolute():
        # Keep root
        parts = list(path_obj.parts)
        if parts[0] == '/':
            safe_parts = ['/']
            parts = parts[1:]
        else:
            safe_parts = []

        # Handle Windows drive letter
        if len(parts) > 0 and ':' in parts[0]:
            safe_parts.append(parts[0])
            parts = parts[1:]
    else:
        parts = list(path_obj.parts)
        safe_parts = []

    # Sanitize remaining parts
    for part in parts:
        if part in ['/', os.sep]:
            safe_parts.append(part)
        else:
            safe_parts.append(sanitize_path_component(part))

    # Join parts and validate
    safe_path = Path(*safe_parts)
    return validate_safe_path(safe_path, root)

@lru_cache(maxsize=1)
def find_project_root() -> Path:
    """Find the project root directory."""
    current_dir = Path(os.path.abspath(__file__)).parent.parent.parent
    while current_dir != current_dir.parent:
        if any((current_dir / marker).exists() for marker in [".git", "pyproject.toml", "README.md"]):
            return current_dir
        current_dir = current_dir.parent
    raise RuntimeError("Unable to determine project root directory")

def project_path(*parts: Union[str, Path]) -> Path:
    """Create an absolute path relative to project root."""
    return PROJECT_ROOT.joinpath(*parts)
