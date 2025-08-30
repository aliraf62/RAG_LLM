"""
core.component_service
======================

Thin facade over :pymod:`core.utils.component_registry` that keeps backward-
compatibility *and* exposes new conveniences.

example:
from core.component_service import get_instance
embedder = get_instance("embedder", "openai")

"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Type

from core.utils.component_registry import (
    available,
    get,
    register,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Registry-based helpers                                                      #
# --------------------------------------------------------------------------- #
def import_component_class(component_path: str) -> Type:
    """Dynamic import based on dotted path `module.Class`."""
    module_path, class_name = component_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def get_instance(category: str, name: str):
    """Return instance from decorator-based runtime registry."""
    return get(category, name)()


def list_available(category: str) -> tuple[str, ...]:
    """List registered names for *category* ('embedder', 'llm', ...)."""
    return available(category)


def register_cli(component_type: str, name: str) -> Callable:
    """Runtime helper to add mapping to registry (legacy compatibility)."""
    def decorator(cls):
        register(component_type, name)(cls)
        return cls

    return decorator


def run_component(component_type: str, component_name: str, args: Dict[str, Any]) -> Any:
    """
    Run a component by type and name with the provided arguments.

    Parameters
    ----------
    component_type : str
        The component category (e.g., "chunker", "cleaner", "exporter")
    component_name : str
        The registered name of the component
    args : Dict[str, Any]
        Arguments to pass to the component

    Returns
    -------
    Any
        The result of running the component

    Raises
    ------
    ValueError
        If the component type or name is not registered
    """
    try:
        # Get the component instance
        component = get_instance(component_type, component_name)

        # Log the execution
        logger.debug(f"Running {component_type}:{component_name} with args: {args}")

        # Different component types may have different execution patterns
        if component_type == "chunker":
            return component.chunk(**args)
        elif component_type == "cleaner":
            return component.clean(**args)
        elif component_type == "exporter":
            return component.export(**args)
        else:
            # Generic execution - try to run a default method or __call__
            if hasattr(component, "run"):
                return component.run(**args)
            elif hasattr(component, "__call__"):
                return component(**args)
            else:
                raise ValueError(f"Component {component_type}:{component_name} has no runnable method")

    except Exception as e:
        logger.error(f"Error running component {component_type}:{component_name}: {str(e)}")
        raise

