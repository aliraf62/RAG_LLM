"""
Core pipeline base classes and data structures.

This module provides the foundational classes and data structures
used throughout the pipeline components, ensuring consistency and
interoperability between loaders, extractors, cleaners, chunkers, etc.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class Row:
    """
    Universal container for document data throughout the pipeline.

    This class serves as the standard data structure passed between
    pipeline components (extractors, cleaners, chunkers, etc.).
    It provides a consistent interface for accessing document content,
    metadata, structured data, and associated assets.

    Attributes
    ----------
    text : str
        The main textual content of the document.
    metadata : Dict[str, Any]
        Document metadata (e.g., title, author, date, source).
    structured : Dict[str, Any]
        Structured representation of the document (e.g., hierarchical content).
    assets : List[str]
        List of asset paths associated with this document.
    id : str, optional
        Unique identifier for this row.
    """
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    structured: Dict[str, Any] = field(default_factory=dict)
    assets: List[str] = field(default_factory=list)
    id: Optional[str] = None

    def __str__(self) -> str:
        """String representation showing text preview and metadata keys."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        metadata_keys = list(self.metadata.keys())
        asset_count = len(self.assets)

        return (f"Row(text='{text_preview}', "
                f"metadata_keys={metadata_keys}, "
                f"assets={asset_count})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the row to a dictionary representation."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "structured": self.structured,
            "assets": self.assets,
            "id": self.id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Row:
        """Create a Row instance from a dictionary."""
        return cls(
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            structured=data.get("structured", {}),
            assets=data.get("assets", []),
            id=data.get("id")
        )


class BasePipelineComponent:
    """
    Base class for all pipeline components.

    This class defines the common interface and functionality
    for all pipeline components (extractors, cleaners, chunkers, etc.).

    Attributes
    ----------
    name : str
        Name of the pipeline component.
    description : str
        Description of what the pipeline component does.
    """

    def __init__(self, name: str = None, description: str = None, **kwargs: Any) -> None:
        """
        Initialize the pipeline component.

        Parameters
        ----------
        name : str, optional
            Name of the component.
        description : str, optional
            Description of what the component does.
        **kwargs : dict
            Additional configuration parameters.
        """
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or f"{self.name} pipeline component"
        self.config = kwargs

    def __str__(self) -> str:
        """String representation showing component name."""
        return f"{self.name}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
