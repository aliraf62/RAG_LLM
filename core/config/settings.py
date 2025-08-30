# core/config/settings.py
"""
Typed, hierarchical configuration for the entire application.

* Loads defaults from `core.config.defaults.DEFAULT_CONFIG`
* Overrides with values read from the project-root `config.yaml`
* Allows optional in-memory overrides (useful for tests)
* Exposes values through a strongly-typed Pydantic model called `AppConfig`
* Provides a global singleton `settings`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent / ".env")

from core.config.defaults import DEFAULT_CONFIG
from core.config.base_paths import find_project_root, PROJECT_ROOT

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

_CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file; return an empty dict if the file is missing/empty."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (override wins)."""
    result: Dict[str, Any] = {**base}
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _env_to_dict() -> Dict[str, Any]:
    """Convert os.environ to a dict, with keys uppercased for compatibility."""
    return {k: v for k, v in os.environ.items()}


# --------------------------------------------------------------------------- #
# Pydantic model                                                              #
# --------------------------------------------------------------------------- #


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # Updated from class Config to model_config

    # ---- OpenAI client configuration ---------------------------------------- #
    client_id: str = Field(default="", description="GenAI client ID for authentication")
    client_secret: str = Field(default="", description="GenAI client secret for authentication")
    sand_token_url: str = Field(default="https://auth.coupa.engineering/v1/token", description="URL for obtaining SAND token")

    # ---- core model/LLM settings ----------------------------------------- #
    model: str = Field(default=DEFAULT_CONFIG["MODEL"])
    vision_model: str = Field(default=DEFAULT_CONFIG["VISION_MODEL"])
    embed_model: str = Field(default=DEFAULT_CONFIG["EMBED_MODEL"])
    default_temperature: float = Field(
        default=DEFAULT_CONFIG["DEFAULT_TEMPERATURE"]
    )
    # ---- retrieval / similarity ----------------------------------------- #
    top_k: int = Field(default=DEFAULT_CONFIG["TOP_K"])
    similarity_threshold: float = Field(
        default=DEFAULT_CONFIG["SIMILARITY_THRESHOLD"]
    )
    filter_low_scores: bool = Field(
        default=DEFAULT_CONFIG["FILTER_LOW_SCORES"]
    )
    # ---- token limits ---------------------------------------------------- #
    max_tokens: int = Field(default=DEFAULT_CONFIG["MAX_TOKENS"])
    max_history_tokens: int = Field(
        default=DEFAULT_CONFIG["MAX_HISTORY_TOKENS"]
    )
    # ---- answer formatting ---------------------------------------------- #
    enable_citations: bool = Field(
        default=DEFAULT_CONFIG["ENABLE_CITATIONS"]
    )
    deduplicate_sources: bool = Field(
        default=DEFAULT_CONFIG["DEDUPLICATE_SOURCES"]
    )
    include_images: bool = Field(default=DEFAULT_CONFIG["INCLUDE_IMAGES"])
    # ---- dict‑like helpers for backward compatibility ------------------- #
    # These methods let legacy code continue using settings["FOO"] or
    # settings.get("FOO", default) until it is fully migrated to attribute
    # access.
    def __getitem__(self, item: str) -> Any:  # noqa: Dunder
        return getattr(self, item)
    # Refactor: safer is to keep settings strictly read-only and for e.g. settigs['blabla']=... to have in-memory mappings but for backward compatibility i keep it now.
    def __setitem__(self, key: str, value: Any) -> None:  # noqa: Dunder
        setattr(self, key, value)

    def get(self, item: str, default: Any | None = None) -> Any:  # noqa: A003
        return getattr(self, item, default)

    def __contains__(self, item: object) -> bool:  # noqa: Dunder
        return hasattr(self, str(item))

    def keys(self):  # noqa: D401
        """Return available config keys (for legacy loops)."""
        return self.__dict__.keys()
    # ---- prompt templates ------------------------------------------------ #
    prompt_templates: Dict[str, str] = Field(
        default_factory=lambda: DEFAULT_CONFIG["PROMPT_TEMPLATES"].copy()
    )


# --------------------------------------------------------------------------- #
# Public loader                                                               #
# --------------------------------------------------------------------------- #


def load_settings(overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
    """
    Build an ``AppConfig`` by merging:

    1.  ``DEFAULT_CONFIG``                         (hard-coded defaults)
    2.  Values from ``config.yaml``                (project-wide overrides)
    3.  Environment variables (.env, shell)        (highest priority)
    4.  *overrides* dict passed in programmatically (tests / cli flags)

    Later items win on conflict.
    """
    yaml_cfg = _load_yaml(_CONFIG_FILE)
    merged = _deep_merge(DEFAULT_CONFIG, yaml_cfg)
    # Merge environment variables (highest priority before overrides)
    env_cfg = _env_to_dict()
    merged = _deep_merge(merged, env_cfg)
    if overrides:
        merged = _deep_merge(merged, overrides)
    # Flatten 'settings' key if present
    if "settings" in merged:
        merged = {**merged, **merged.pop("settings")}
    return AppConfig(**merged)


# --------------------------------------------------------------------------- #
# Global singleton, initialized immediately                                   #
# --------------------------------------------------------------------------- #

settings: AppConfig = load_settings()


# --------------------------------------------------------------------------- #
# Apply customer settings without circular imports                            #
# --------------------------------------------------------------------------- #

def apply_customer_settings(customer_id: str) -> None:
    """
    Apply customer-specific settings to the global settings object.
    
    This should only be called after the customer's modules have been loaded.
    
    Args:
        customer_id: Customer identifier
    """
    # Import here to avoid circular import
    from core.services.customer_service import customer_service

    # Get customer settings
    customer_config = customer_service.get_customer_settings(customer_id)

    if not customer_config:
        return

    # Extract settings from customer config
    customer_settings = customer_config.get("settings", {})
    
    # Log what we're doing for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Applying customer settings for {customer_id}")
    logger.debug(f"Customer settings: {customer_settings}")

    # Apply settings to the global settings object
    global settings

    # Deep merge the customer settings into the current settings
    current_settings = settings.model_dump()
    merged_settings = _deep_merge(current_settings, customer_settings)
    
    # Handle special configuration sections that might need separate handling
    # For example, datasets might need to be handled differently
    if "datasets" in customer_config and "datasets" not in customer_settings:
        logger.info("Found datasets configuration outside of settings key, merging separately")
        if "datasets" not in merged_settings:
            merged_settings["datasets"] = {}
        merged_settings["datasets"] = _deep_merge(
            merged_settings["datasets"], 
            customer_config["datasets"]
        )
    
    settings = AppConfig(**merged_settings)
