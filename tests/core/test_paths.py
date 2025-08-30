"""Tests for core path handling functionality."""
from pathlib import Path
import pytest
from unittest.mock import patch

from core.config.base_paths import (
    PROJECT_ROOT,
    CUSTOMERS_DIR,
    find_project_root,
    project_path
)

from core.config.paths import (
    get_customer_root,
    get_customer_outputs_dir,
    get_customer_assets_dir,
    get_customer_vector_store_dir,
    ensure_customer_dirs
)

from core.config.settings import settings

@pytest.fixture
def mock_settings(monkeypatch):
    """Fixture providing test settings."""
    test_settings = {
        "PATHS": {
            "OUTPUTS_ROOT": "outputs",
            "VECTOR_STORE_ROOT": "vector_store",
            "CUSTOMERS_ROOT": "customers",
            "DATASETS_ROOT": "datasets",
            "ASSETS_ROOT": "assets"
        },
        "PATH_SCOPES": {
            "VECTORSTORE_SCOPE": "dataset",
            "ASSETS_SCOPE": "customer",
            "OUTPUTS_SCOPE": "dataset"
        }
    }
    def mock_get(key, default=None):
        return test_settings.get(key, default)
    monkeypatch.setattr(settings, "get", mock_get)
    return test_settings

def test_find_project_root():
    """Test project root detection."""
    root = find_project_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()

def test_project_path():
    """Test project path resolution."""
    test_path = project_path("test", "path")
    assert test_path == PROJECT_ROOT / "test" / "path"

def test_get_customer_root():
    """Test customer root path resolution."""
    with patch('core.config.base_paths.PROJECT_ROOT', Path('/test')):
        customer_root = get_customer_root("test_customer")
        assert customer_root == Path('/test/customers/test_customer')

def test_customer_outputs_dir(mock_settings):
    """Test outputs directory with scope handling."""
    with patch('core.config.base_paths.PROJECT_ROOT', Path('/test')):
        # Test dataset-scoped output
        outputs_dir = get_customer_outputs_dir("test_customer", "test_dataset")
        assert outputs_dir == Path('/test/customers/test_customer/outputs/test_dataset')

def test_customer_assets_dir(mock_settings):
    """Test assets directory with scope handling."""
    with patch('core.config.base_paths.PROJECT_ROOT', Path('/test')):
        # Test customer-scoped assets
        assets_dir = get_customer_assets_dir("test_customer")
        assert assets_dir == Path('/test/customers/test_customer/assets')

        # Test dataset-scoped assets
        assets_dir = get_customer_assets_dir("test_customer", "test_dataset")
        assert assets_dir == Path('/test/customers/test_customer/assets/test_dataset')

def test_vector_store_dir(mock_settings):
    """Test vector store directory with scope handling."""
    with patch('core.config.base_paths.PROJECT_ROOT', Path('/test')):
        # Test dataset-scoped vector store
        vector_dir = get_customer_vector_store_dir("test_customer", "test_dataset")
        assert vector_dir == Path('/test/customers/test_customer/vector_store/test_dataset')

        # Test customer-scoped vector store
        with patch.dict(mock_settings["PATH_SCOPES"], {"VECTORSTORE_SCOPE": "customer"}):
            vector_dir = get_customer_vector_store_dir("test_customer")
            assert vector_dir == Path('/test/customers/test_customer/vector_store/all')

def test_directory_creation(tmp_path, mock_settings):
    """Test directory creation with proper permissions."""
    with patch('core.config.base_paths.PROJECT_ROOT', tmp_path):
        # Test customer directories
        customer_dir = get_customer_root("test_customer")
        ensure_customer_dirs("test_customer")
        assert customer_dir.exists()

        # Verify core directories exist
        assert (customer_dir / "outputs").exists()
        assert (customer_dir / "vector_store").exists()
        assert (customer_dir / "datasets").exists()
        assert (customer_dir / "config").exists()
        assert (customer_dir / "temp").exists()

def test_invalid_customer_id():
    """Test handling of invalid customer IDs."""
    with pytest.raises(ValueError):
        get_customer_root("")
    with pytest.raises(ValueError):
        get_customer_root(None)

def test_config_hierarchy(mock_settings):
    """Test configuration hierarchy is respected."""
    # Test that paths respect settings overrides
    with patch.dict(mock_settings["PATHS"], {"OUTPUTS_ROOT": "custom_outputs"}):
        outputs_dir = get_customer_outputs_dir("test_customer")
        assert "custom_outputs" in str(outputs_dir)

    # Test that customer config overrides work
    customer_settings = {"PATHS": {"VECTOR_STORE_ROOT": "custom_vectors"}}
    with patch("core.config.settings.settings.get", return_value=customer_settings):
        vector_dir = get_customer_vector_store_dir("test_customer")
        assert "custom_vectors" in str(vector_dir)
