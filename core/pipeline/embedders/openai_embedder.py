# pipeline/embedders/openai_embedder.py

"""
OpenAI embedding provider with optional async support.
"""
from __future__ import annotations
import logging
from typing import List, Sequence

from core.pipeline import Row
from core.utils.component_registry import register
from core.utils.exceptions import EmbeddingError
from core.llm import get_llm_client
from core.config.settings import settings
from core.pipeline.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

@register("embedder", "openai")
class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder using OpenAI’s embedding API, sync or async based on settings.

    Respects:
      - settings.EMBED_MODEL
      - settings.EMBEDDER_ASYNC (bool): default True
    """
    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.get("EMBED_MODEL")
        self._provider = get_llm_client("openai")  # thread-safe memoised client

    def embed_text(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:
        """
        Return embeddings for texts. Only synchronous API is supported.
        If the OpenAI API cannot handle batch requests, fall back to serial embedding with retry.
        """
        inputs = list(texts)
        # Print debug info for the first and last line of each chunk (not the whole chunk)
        for i, t in enumerate(inputs):
            lines = t.splitlines()
            preview = lines[0][:120] if lines else t[:120]
            if len(lines) > 1:
                preview += ' ... ' + lines[-1][:120]
            print(f"[EMBED_DEBUG] Chunk {i}: {preview}")
            if i >= 2:
                print(f"[EMBED_DEBUG] ... ({len(inputs)-3} more chunks omitted) ...")
                break
        try:
            # Try batch embedding first
            vectors = self._provider.get_embeddings(inputs, model=self.model)
            return vectors
        except Exception as exc:
            logger.warning("Batch embedding failed, falling back to serial embedding. Error: %s", exc)
            # Fallback: serial embedding with retry
            vectors = []
            for t in inputs:
                backoff = 1
                while True:
                    try:
                        v = self._provider.get_embeddings([t], model=self.model)[0]
                        vectors.append(v)
                        break
                    except Exception as exc2:
                        logger.error("Serial embedding failed for input: %s, error: %s", t[:80], exc2)
                        import time
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 10)
            return vectors

    def embed_rows(self, rows: Sequence[Row]) -> List[List[float]]:
        """
        Embed a list of Row objects into vectors.

        Parameters
        ----------
        rows : Sequence[Row]
            The input Row objects to embed.

        Returns
        -------
        List[List[float]]
            A list of embedding vectors.
        """
        # Extract text from Row objects
        texts = [row.text for row in rows]

        # Use the existing embed_text method
        return self.embed_text(texts)

