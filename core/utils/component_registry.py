"""
core.utils.component_registry
=============================

Single authoritative plugin registry.

* Register:   ``@register("chunker", "markdown")``
* Discover:   ``cls = get("chunker", "markdown")``
* Enumerate:  ``available("chunker")  ->  ("html", "markdown", "text")``

No legacy CLI mapping, no extra wrapper classes – one system only.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# {category: {name: factory}}
_REGISTRY: Dict[str, Dict[str, Callable[[], Any]]] = defaultdict(dict)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def register(category: str, name: str):
    """
    Decorator for registering a factory under *category* / *name*.

    Example
    -------
    ```python
    @register("chunker", "text")
    class TextChunker(BaseChunker):
        ...
    ```
    """

    def decorator(factory: Callable[[], Any] | Callable):
        _REGISTRY[category][name] = factory
        logger.debug("Registered %s:%s", category, name)
        return factory

    return decorator


def get(category: str, name: str):
    """Return a **class or factory** registered as *category:name*."""
    if category not in _REGISTRY:
        raise KeyError(f"No such category registered: {category!r}")
    if name not in _REGISTRY[category]:
        raise KeyError(f"No {category!r} named {name!r} available.")
    return _REGISTRY[category][name]


def available(category: str) -> Tuple[str, ...]:
    """Return the sorted, frozen list of available names for *category*."""
    return tuple(sorted(_REGISTRY.get(category, {}).keys()))


# --------------------------------------------------------------------------- #
# Component Message Helpers                                                   #
# --------------------------------------------------------------------------- #
def component_message(message_key: str, component_type: str = None, component_name: str = None, **kwargs):
    """
    Get a component-related message with component type and name automatically included.

    This makes it easy to get properly formatted messages for components without
    having to manually specify the component_type and component_name each time.

    Args:
        message_key: The message identifier
        component_type: Optional component type override
        component_name: Optional component name override
        **kwargs: Additional format variables

    Returns:
        The translated and formatted message
    """
    from core.utils.i18n import get_message

    format_args = kwargs.copy()
    if component_type:
        format_args["component_type"] = component_type
    if component_name:
        format_args["component_name"] = component_name

    return get_message(message_key, **format_args)


def register_cli_component(component_type: str, name: str):
    """
    Register a CLI component.

    Args:
        component_type: Type of component (e.g., "command", "exporter")
        name: Name to register under
    """

    def decorator(cls):
        register(component_type, name)(cls)
        return cls

    return decorator


def create_component_instance(category: str, name: str, **kwargs):
    """
    Retrieve and instantiate a component from the registry.
    Raises informative error if not found.
    """
    from core.utils.component_registry import get, available
    try:
        cls = get(category, name)
    except KeyError:
        raise ValueError(
            f"No {category!r} named {name!r} registered. "
            f"Available: {available(category)}"
        )
    return cls(**kwargs)