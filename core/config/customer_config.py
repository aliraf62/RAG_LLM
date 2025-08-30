"""
core.config.customer_config
==========================

Customer configuration management system.

Provides functionality to load, validate, and access customer-specific configuration
using a standardized directory structure and configuration hierarchy.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator, Union

import yaml

from core.config.settings import settings
from core.config.paths import customer_path, PROJECT_ROOT
from core.utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class CustomerConfigManager:
    """
    Customer configuration manager.

    Loads and provides access to customer configuration files following
    a standardized directory structure with separate files for different domains.
    Implements the configuration hierarchy where method parameters override
    customer config, which overrides global config, which overrides defaults.
    """

    def __init__(self, customer_id: str):
        """
        Initialize with a customer ID.

        Parameters
        ----------
        customer_id : str
            Customer identifier
        """
        self.customer_id = customer_id
        self.customer_root = customer_path(customer_id)
        self.main_config = self._load_main_config()

        # Cache for loaded configurations
        self._dataset_configs: Dict[str, Dict[str, Any]] = {}
        self._pipeline_configs: Dict[str, Dict[str, Any]] = {}
        self._schema_configs: Dict[str, Dict[str, Any]] = {}

    def _load_main_config(self) -> Dict[str, Any]:
        """
        Load the main customer configuration file.

        Returns
        -------
        Dict[str, Any]
            Main customer configuration

        Raises
        ------
        ConfigurationError
            If the customer configuration file doesn't exist or is invalid
        """
        config_file = self.customer_root / f"{self.customer_id}.yaml"

        if not config_file.exists():
            raise ConfigurationError(f"Customer configuration not found: {config_file}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                logger.debug(f"Loaded main config for customer {self.customer_id}")
                return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load customer configuration: {e}")

    def get_config_path(self, config_type: str) -> Path:
        """
        Get the path to a specific configuration directory.

        Parameters
        ----------
        config_type : str
            Type of configuration (datasets, pipelines, schemas)

        Returns
        -------
        Path
            Path to the configuration directory
        """
        # Get standard path from settings
        path_key = f"{config_type.upper()}_DIR"
        standard_path = settings.get("CUSTOMER_CONFIG_PATHS", {}).get(path_key)

        if not standard_path:
            raise ConfigurationError(f"Unknown configuration type: {config_type}")

        return self.customer_root / standard_path

    def get_dataset_config(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier

        Returns
        -------
        Dict[str, Any]
            Dataset configuration

        Raises
        ------
        ConfigurationError
            If the dataset configuration doesn't exist or is invalid
        """
        # Return cached config if available
        if dataset_id in self._dataset_configs:
            return self._dataset_configs[dataset_id]

        # Try to load from the new structured config
        dataset_dir = self.get_config_path("datasets")
        dataset_file = dataset_dir / f"{dataset_id}.yaml"

        # Check if dataset file exists in the new structure
        if dataset_file.exists():
            try:
                with open(dataset_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                    self._dataset_configs[dataset_id] = config
                    logger.debug(f"Loaded dataset config for {dataset_id} from {dataset_file}")
                    return config
            except Exception as e:
                raise ConfigurationError(f"Failed to load dataset configuration: {e}")

        # Fall back to legacy format in main config
        datasets = self.main_config.get("datasets", {})
        if dataset_id in datasets:
            self._dataset_configs[dataset_id] = datasets[dataset_id]
            logger.debug(f"Loaded dataset config for {dataset_id} from main config")
            return datasets[dataset_id]

        raise ConfigurationError(f"Dataset configuration not found: {dataset_id}")

    def get_pipeline_config(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific pipeline.

        Parameters
        ----------
        pipeline_id : str
            Pipeline identifier (e.g., extraction, rag)

        Returns
        -------
        Dict[str, Any]
            Pipeline configuration

        Raises
        ------
        ConfigurationError
            If the pipeline configuration doesn't exist or is invalid
        """
        # Return cached config if available
        if pipeline_id in self._pipeline_configs:
            return self._pipeline_configs[pipeline_id]

        # Try to load from the new structured config
        pipeline_dir = self.get_config_path("pipelines")
        pipeline_file = pipeline_dir / f"{pipeline_id}.yaml"

        # Check if pipeline file exists in the new structure
        if pipeline_file.exists():
            try:
                with open(pipeline_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                    self._pipeline_configs[pipeline_id] = config
                    logger.debug(f"Loaded pipeline config for {pipeline_id} from {pipeline_file}")
                    return config
            except Exception as e:
                raise ConfigurationError(f"Failed to load pipeline configuration: {e}")

        raise ConfigurationError(f"Pipeline configuration not found: {pipeline_id}")

    def get_available_datasets(self) -> List[str]:
        """
        Get a list of available dataset IDs for this customer.

        Returns
        -------
        List[str]
            List of dataset IDs
        """
        datasets = []

        # Check datasets from main config
        config_datasets = self.main_config.get("config", {}).get("datasets", [])
        if config_datasets:
            datasets.extend(config_datasets)

        # Also check for dataset files in the config directory
        dataset_dir = self.get_config_path("datasets")
        if dataset_dir.exists():
            for dataset_file in dataset_dir.glob("*.yaml"):
                dataset_id = dataset_file.stem
                if dataset_id not in datasets:
                    datasets.append(dataset_id)

        return datasets

    def get_dataset_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configurations for all available datasets.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping dataset IDs to their configurations
        """
        result = {}

        for dataset_id in self.get_available_datasets():
            try:
                result[dataset_id] = self.get_dataset_config(dataset_id)
            except ConfigurationError as e:
                logger.warning(f"Failed to load config for dataset {dataset_id}: {e}")

        return result

    def get_dataset_provider(self, dataset_id: str, component_type: str) -> str:
        """
        Get the provider for a specific component of a dataset's pipeline.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        component_type : str
            Component type (loader, extractor, chunker, etc.)

        Returns
        -------
        str
            Provider identifier

        Raises
        ------
        ConfigurationError
            If the dataset or component configuration doesn't exist
        """
        try:
            # Get the dataset configuration
            dataset_config = self.get_dataset_config(dataset_id)

            # Check the dataset's pipeline configuration for the component
            pipeline_config = dataset_config.get("pipeline", {})
            if component_type in pipeline_config:
                return pipeline_config[component_type]

            # If not found, check component-specific configuration
            components_config = dataset_config.get("components", {})
            component_config = components_config.get(component_type, {})
            if "provider" in component_config:
                return component_config["provider"]

            # If still not found, fall back to customer-level settings
            return settings.get(f"{component_type.upper()}_PROVIDER", "default")

        except Exception as e:
            logger.error(f"Error getting dataset provider: {e}")
            return "default"


# Singleton pattern for customer config managers
_customer_config_managers: Dict[str, CustomerConfigManager] = {}

def get_customer_config(customer_id: str) -> CustomerConfigManager:
    """
    Get or create a customer configuration manager for a specific customer.

    Parameters
    ----------
    customer_id : str
        Customer identifier

    Returns
    -------
    CustomerConfigManager
        Customer configuration manager instance

    Raises
    ------
    ConfigurationError
        If the customer configuration doesn't exist or is invalid
    """
    if customer_id not in _customer_config_managers:
        _customer_config_managers[customer_id] = CustomerConfigManager(customer_id)

    return _customer_config_managers[customer_id]
