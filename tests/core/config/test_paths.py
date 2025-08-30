"""Tests for path handling functionality."""
from pathlib import Path
import pytest
from unittest.mock import patch

from core.config.base_paths import (
    PROJECT_ROOT,
    CUSTOMERS_DIR,
    SHARED_DIR,
    find_project_root,
    project_path
)

from core.config.paths import (
    get_customer_root,
    get_customer_outputs_dir,
    get_customer_assets_dir,
    get_customer_vector_store_dir,
    get_customer_datasets_dir,
    get_customer_config_dir,
    ensure_customer_dirs
)

def test_customer_root():
    """Test customer root path resolution."""
    with patch("core.config.base_paths.PROJECT_ROOT", Path("/test")):
        path = get_customer_root("test_customer")
        assert path == Path("/test/customers/test_customer")

        # Test invalid customer IDs
        with pytest.raises(ValueError):
            get_customer_root("")
        with pytest.raises(ValueError):
            get_customer_root(None)

def test_customer_outputs():
    """Test customer outputs directory structure."""
    with patch("core.config.base_paths.PROJECT_ROOT", Path("/test")):
        # Test without dataset
        path = get_customer_outputs_dir("test_customer")
        assert path == Path("/test/customers/test_customer/outputs")

        # Test with dataset
        path = get_customer_outputs_dir("test_customer", "test_dataset")
        assert path == Path("/test/customers/test_customer/outputs/test_dataset")

def test_vector_store():
    """Test vector store directory structure."""
    with patch("core.config.base_paths.PROJECT_ROOT", Path("/test")):
        # Test dataset-specific store
        path = get_customer_vector_store_dir("test_customer", "test_dataset", "faiss")
        assert path == Path("/test/customers/test_customer/vector_store/test_dataset_faiss")

        # Test combined store
        path = get_customer_vector_store_dir("test_customer", backend="faiss")
        assert path == Path("/test/customers/test_customer/vector_store/all_datasets_faiss")

def test_assets():
    """Test assets directory structure."""
    with patch("core.config.base_paths.PROJECT_ROOT", Path("/test")):
        # Test customer-level assets
        path = get_customer_assets_dir("test_customer")
        assert path == Path("/test/customers/test_customer/outputs/assets")

        # Test dataset-specific assets
        path = get_customer_assets_dir("test_customer", "test_dataset")
        assert path == Path("/test/customers/test_customer/outputs/test_dataset/assets")

def test_customer_datasets():
    """Test datasets directory."""
    with patch("core.config.base_paths.PROJECT_ROOT", Path("/test")):
        path = get_customer_datasets_dir("test_customer")
        assert path == Path("/test/customers/test_customer/datasets")

def test_customer_config():
    """Test config directory."""
    with patch("core.config.base_paths.PROJECT_ROOT", Path("/test")):
        path = get_customer_config_dir("test_customer")
        assert path == Path("/test/customers/test_customer/config")

def test_ensure_dirs(tmp_path):
    """Test directory creation."""
    with patch("core.config.base_paths.PROJECT_ROOT", tmp_path):
        ensure_customer_dirs("test_customer")

        # Check all required directories were created
        customer_root = tmp_path / "customers" / "test_customer"
        assert (customer_root / "outputs").exists()
        assert (customer_root / "vector_store").exists()
        assert (customer_root / "assets").exists()
        assert (customer_root / "datasets").exists()
        assert (customer_root / "config").exists()

        # Check they're actually directories
        assert (customer_root / "outputs").is_dir()

def test_project_root_detection():
    """Test project root detection works."""
    root = find_project_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()

def test_project_path():
    """Test project path resolution."""
    with patch("core.config.base_paths.PROJECT_ROOT", Path("/test")):
        path = project_path("foo", "bar")
        assert path == Path("/test/foo/bar")
