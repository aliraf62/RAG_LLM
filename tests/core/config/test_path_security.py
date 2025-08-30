"""Tests for path security validation."""
import os

import pytest

from core.config.base_paths import (
    validate_safe_path,
    enforce_path_safety,
    sanitize_path_component,
    validate_customer_id,
    PathValidationError,
    PROJECT_ROOT
)


def test_sanitize_path_component():
    """Test path component sanitization."""
    # Test valid components
    assert sanitize_path_component("valid-name") == "valid-name"
    assert sanitize_path_component("test_123") == "test_123"

    # Test components needing sanitization
    assert sanitize_path_component("test$file") == "testfile"

    # Test invalid components
    with pytest.raises(PathValidationError):
        sanitize_path_component("../bad")
    with pytest.raises(PathValidationError):
        sanitize_path_component("\\bad")
    with pytest.raises(PathValidationError):
        sanitize_path_component("")

def test_validate_safe_path():
    """Test path validation."""
    # Test valid paths
    test_root = PROJECT_ROOT / "test_dir"
    safe_path = test_root / "safe" / "path.txt"

    # Should pass validation
    validated = validate_safe_path(safe_path, test_root)
    assert validated.is_absolute()
    assert str(validated).startswith(str(test_root))

    # Test path traversal attempts
    with pytest.raises(PathValidationError):
        validate_safe_path(test_root / ".." / "escape.txt")

    with pytest.raises(PathValidationError):
        validate_safe_path(test_root / "sub" / ".." / ".." / "escape.txt")

    # Test absolute path outside root
    with pytest.raises(PathValidationError):
        validate_safe_path("/etc/passwd", test_root)

def test_enforce_path_safety():
    """Test path safety enforcement."""
    test_root = PROJECT_ROOT / "test_dir"

    # Test basic path
    safe_path = enforce_path_safety("test/path.txt", test_root)
    assert safe_path.is_absolute()
    assert str(safe_path).startswith(str(test_root))

    # Test path with unsafe characters
    unsafe_path = "test/bad$path../file.txt"
    safe_path = enforce_path_safety(unsafe_path, test_root)
    assert "$" not in str(safe_path)
    assert ".." not in str(safe_path)

    # Test complete path traversal rejection
    with pytest.raises(PathValidationError):
        enforce_path_safety("../../../etc/passwd", test_root)

def test_validate_customer_id():
    """Test customer ID validation."""
    # Test valid IDs
    assert validate_customer_id("valid-customer") == "valid-customer"
    assert validate_customer_id("test123") == "test123"
    assert validate_customer_id("customer_name") == "customer_name"

    # Test invalid IDs
    with pytest.raises(PathValidationError):
        validate_customer_id("")  # Empty

    with pytest.raises(PathValidationError):
        validate_customer_id("invalid/customer")  # Contains slash

    with pytest.raises(PathValidationError):
        validate_customer_id("customer@company")  # Contains @

    with pytest.raises(PathValidationError):
        validate_customer_id("." * 65)  # Too long

def test_path_length_limit():
    """Test path length limits are enforced."""
    test_root = PROJECT_ROOT / "test_dir"
    long_name = "x" * 300  # Exceeds MAX_PATH_LENGTH

    with pytest.raises(PathValidationError):
        validate_safe_path(test_root / long_name)

    with pytest.raises(PathValidationError):
        enforce_path_safety(long_name, test_root)

def test_windows_reserved_names():
    """Test Windows reserved names are blocked."""
    test_root = PROJECT_ROOT / "test_dir"

    for reserved in ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]:
        with pytest.raises(PathValidationError):
            validate_safe_path(test_root / reserved)

        with pytest.raises(PathValidationError):
            validate_safe_path(test_root / "sub" / reserved)

@pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
def test_windows_device_paths():
    """Test Windows device paths are blocked."""
    with pytest.raises(PathValidationError):
        validate_safe_path(r"\\.\COM1")

    with pytest.raises(PathValidationError):
        validate_safe_path(r"\\?\C:\Windows\System32")
