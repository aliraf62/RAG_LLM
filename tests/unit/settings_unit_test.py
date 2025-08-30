import types
import pytest
from core.config.settings import settings, AppConfig

def test_settings_singleton_is_appconfig_instance():
    assert isinstance(settings, AppConfig)

def test_dict_like_access():
    # attribute
    assert settings.model == settings["model"]
    # .get fallback
    assert settings.get("non_existing_key", 42) == 42

def test_immutability_of_singleton():
    with pytest.raises(AttributeError):
        settings.new_key = "value"          # type: ignore[attr-defined]

def test_defaults_loaded():
    # value defined in DEFAULT_CONFIG must exist
    assert hasattr(settings, "max_tokens")