# pipeline/loaders/base.py
"""
Abstract document loader that yields Row objects.

A concrete loader only needs to implement `_iter_sources()`, yielding
(path, text, metadata_dict) tuples. The base class converts each tuple to a
standardized `Row` object.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, Tuple, Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod

from core.utils.models import Row
from core.pipeline.base import BasePipelineComponent

logger = logging.getLogger(__name__)


class BaseLoader(BasePipelineComponent, ABC):
    """
    Abstract iterable loader for document sources.

    Base class for all loaders that provides common functionality
    and standardized Row object creation.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the loader.

        Parameters
        ----------
        **kwargs : Any
            Additional configuration options
        """
        super().__init__(name=kwargs.pop("name", self.__class__.__name__), **kwargs)

    @abstractmethod
    def load(
        self,
        source: Union[str, Path, Iterable[Dict[str, object]]],
    ) -> Iterable[Row]:
        """
        Load documents from source.

        Parameters
        ----------
        source : Path | str | Iterable[Dict[str, object]]
            Source to load documents from, can be a path or in-memory data

        Returns
        -------
        Iterable[Row]
            Loaded documents as Row objects
        """
        pass

    def __iter__(self) -> Iterator[Row]:
        """
        Iterate through all sources and yield Row objects.

        Returns
        -------
        Iterator[Row]
            Iterator of standardized Row objects
        """
        for path, text, metadata, structured, assets in self._iter_sources():
            # Ensure path is converted to string and added to metadata
            metadata = dict(metadata) | {"source_path": str(path)}

            yield Row(
                text=text,
                metadata=metadata,
                structured=structured,
                assets=assets
            )

    @abstractmethod
    def _iter_sources(self) -> Iterator[Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]]:
        """
        Iterate through document sources, yielding path, text and metadata.

        Each concrete loader must implement this method to provide its
        specific loading logic.

        Yields
        ------
        Tuple[Path, str, Dict[str, Any], Dict[str, Any], List[str]]
            Tuples of (path, text, metadata, structured_data, assets)
        """
        pass

