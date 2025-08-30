"""
core.services.customer_service
==============================

Manages customer-specific components and configurations.

* Discovers and loads customer-specific components
* Manages customer-specific configurations
* Provides customer context for the application
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from core.config.paths import project_path, customer_path
from core.utils.component_registry import _REGISTRY

logger = logging.getLogger(__name__)

class CustomerService:
    """Manages customer-specific components and configurations."""

    def __init__(self):
        self._loaded_customers: Set[str] = set()
        self._active_customer: Optional[str] = None
        self._customer_configs: Dict[str, Dict[str, Any]] = {}

    @property
    def active_customer(self) -> Optional[str]:
        """Return the currently active customer ID."""
        return self._active_customer

    def list_available_customers(self) -> List[str]:
        """Return a list of available customer directories."""
        customers_dir = project_path("customers")

        if not customers_dir.exists() or not customers_dir.is_dir():
            return []

        return [
            d.name for d in customers_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}.yaml").exists()
        ]

    def load_customer(self, customer_id: str, activate: bool = True) -> bool:
        """
        Load a customer's components and configuration.

        Args:
            customer_id: The customer identifier
            activate: Whether to set this customer as active

        Returns:
            True if customer was successfully loaded, False otherwise
        """
        if customer_id in self._loaded_customers:
            if activate:
                self._active_customer = customer_id
            return True

        customer_dir = customer_path(customer_id)

        if not customer_dir.exists() or not customer_dir.is_dir():
            logger.error(f"Customer directory not found: {customer_dir}")
            return False

        config_path = customer_dir / f"{customer_id}.yaml"
        if not config_path.exists():
            logger.error(f"Customer configuration file not found: {config_path}")
            return False

        # Load customer configuration
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                self._customer_configs[customer_id] = config
        except Exception as e:
            logger.error(f"Failed to load customer configuration: {e}")
            return False

        # Import customer python modules to trigger registration
        self._import_customer_modules(customer_id)

        self._loaded_customers.add(customer_id)
        if activate:
            self._active_customer = customer_id

        logger.info(f"Loaded customer: {customer_id}")
        return True

    def get_customer_config(self, customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific customer or the active customer."""
        customer_id = customer_id or self._active_customer

        if not customer_id or customer_id not in self._loaded_customers:
            return {}

        return self._customer_configs.get(customer_id, {})

    def get_customer_settings(self, customer_id: str) -> Dict[str, Any]:
        """
        Load customer-specific settings from the customer's YAML file.

        Parameters
        ----------
        customer_id : str
            Customer identifier

        Returns
        -------
        Dict[str, Any]
            Customer-specific settings
        """
        # If we've already loaded this customer's config, return it
        if customer_id in self._customer_configs:
            return self._customer_configs[customer_id]

        config_path = customer_path(customer_id) / f"{customer_id}.yaml"
        if not config_path.exists():
            logger.warning(f"No config file found for customer {customer_id} at {config_path}")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                customer_config = yaml.safe_load(f) or {}

            # Cache the config for future use
            self._customer_configs[customer_id] = customer_config
            return customer_config
        except Exception as e:
            logger.error(f"Error loading config for customer {customer_id}: {e}")
            return {}

    def _import_customer_modules(self, customer_id: str) -> None:
        """Import Python modules from customer directory to trigger registration."""
        customer_dir = customer_path(customer_id)

        # Define directories to scan for Python modules
        module_dirs = [
            customer_dir / "extractors",
            customer_dir / "exporters",
            customer_dir / "loaders",
            customer_dir / "chunkers",
        ]

        for module_dir in module_dirs:
            if not module_dir.exists() or not module_dir.is_dir():
                continue

            for py_file in module_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                module_name = f"customers.{customer_id}.{module_dir.name}.{py_file.stem}"

                try:
                    # Import the module to trigger registration decorators
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        logger.debug(f"Imported customer module: {module_name}")
                except Exception as e:
                    logger.error(f"Failed to import customer module {module_name}: {e}")

    def get_registered_components(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently registered components."""
        return _REGISTRY

    def get_customer_components(self, customer_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get components registered for a specific customer.

        This is a best-effort function that tries to identify customer-specific
        components based on module path.
        """
        customer_id = customer_id or self._active_customer
        if not customer_id:
            return {}

        result = {}

        for category, components in _REGISTRY.items():
            customer_components = {}

            for name, factory in components.items():
                # Try to determine if this component belongs to the customer
                # based on its module path
                module = factory.__module__
                if module.startswith(f"customers.{customer_id}"):
                    customer_components[name] = factory

            if customer_components:
                result[category] = customer_components

        return result

# Singleton instance
customer_service = CustomerService()
