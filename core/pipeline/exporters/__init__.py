from __future__ import annotations
"""
pipeline/exporters/__init__.py

Exporters output processed data to various destinations.
All concrete exporters register themselves under the "exporter" category.
"""

from core.utils.component_registry import (
    register as register_exporter,
    get as _get,
    available as _available,
)

from .base import BaseExporter
# Auto-register concrete implementations on import:
# from .<your_exporter> import <YourExporter>  # noqa: F401

from typing import Type, Tuple

__all__ = [
    "BaseExporter",
    "register_exporter",
    "get_exporter",
    "list_available_exporters",
    "create_exporter",
]


def get_exporter(name: str) -> Type[BaseExporter]:
    """
    Retrieve the exporter *class* registered under `name`.

    Parameters
    ----------
    name : str
        Registry key of the exporter.

    Returns
    -------
    Type[BaseExporter]
        The exporter class itself.
    """
    return _get("exporter", name)  # type: ignore[return-value]


def list_available_exporters() -> Tuple[str, ...]:
    """
    List all registered exporter names.

    Returns
    -------
    Tuple[str, ...]
        Registry keys of all available exporters.
    """
    return _available("exporter")


def create_exporter(name: str, **cfg) -> BaseExporter:
    """
    Instantiate an exporter by its registry name, passing any config.

    Parameters
    ----------
    name : str
        Registry key of the exporter.
    **cfg
        Keyword args forwarded to the exporter's __init__.

    Returns
    -------
    BaseExporter
        An instance of the requested exporter.
    """
    ExporterCls = get_exporter(name)
    return ExporterCls(**cfg)