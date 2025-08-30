"""
core.config.service
==========================

Centralised configuration loader that merges, in strict precedence:

1. Hard‑coded defaults from :pymod:`core.settings`
2. Root‑level project YAML(`config.yaml`)
3. Customer‑specific YAML(`customers/<id>/config/<id>.yaml`)
4. Runtime overrides supplied by the caller

The resulting dict is returned by :py:meth:`ConfigurationService.get_config`.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from core.config.settings import settings, _deep_merge as deep_merge_dicts


class ConfigurationContext:
    """
    Simple value‑object passed by callers to indicate *which* customer/user
    the effective configuration should be computed for.

    Only `customer_id` is used right now, but `user_id` is kept for future
    per‑user overrides.
    """

    def __init__(self, customer_id: str | None = None, user_id: str | None = None):
        self.customer_id = customer_id
        self.user_id = user_id  # reserved for future use


class ConfigurationService:
    """
    Public façade used by the rest of the codebase.

    Instantiate once and reuse – all YAML files are cached after the first
    load, so repeated `get_config` calls are cheap.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        project_root: Path | None = None,
        user_overrides: Dict[str, Any] | None = None,
    ) -> None:
        self._project_root = project_root or Path(__file__).resolve().parents[2]
        self._logger = logging.getLogger(self.__class__.__name__)
        self._user_overrides: Dict[str, Any] = user_overrides or {}

        # Cache global YAML once
        self._global_yaml_cfg: Dict[str, Any] = self._load_yaml(
            self._project_root / "config.yaml"
        )

        # Customer YAML cache – {customer_id: cfg_dict}
        self._customer_cfg_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def get_config(self, context: ConfigurationContext | None = None) -> Dict[str, Any]:
        """
        Return an **effective** configuration dict for *context*.

        If *context* is ``None`` or has no ``customer_id``, only defaults +
        root YAML + runtime overrides are applied.
        """
        context = context or ConfigurationContext()

        customer_cfg: Dict[str, Any] = {}
        if context.customer_id:
            customer_cfg = self._get_customer_cfg(context.customer_id)

        merged = deep_merge_dicts(
            deep_merge_dicts(
                deep_merge_dicts(settings.dict(), self._global_yaml_cfg),
                customer_cfg
            ),
            self._user_overrides
        )
        return merged

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Return a dict for YAML file *path* or empty dict if missing/invalid."""
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            self._logger.error("Failed to parse YAML %s: %s", path, exc)
            return {}

    def _get_customer_cfg(self, customer_id: str) -> Dict[str, Any]:
        """Load (and cache) customer YAML."""
        if customer_id in self._customer_cfg_cache:
            return self._customer_cfg_cache[customer_id]

        cfg_path = (
            self._project_root
            / "customers"
            / customer_id
            / "config"
            / f"{customer_id}.yaml"
        )
        cfg_dict = self._load_yaml(cfg_path)
        self._customer_cfg_cache[customer_id] = cfg_dict
        return cfg_dict