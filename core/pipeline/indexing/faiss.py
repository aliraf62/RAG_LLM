"""
FAISS backend – registers as “faiss”.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Callable

import faiss
import numpy as np
from langchain.schema import Document

from core.utils.component_registry import register
from core.utils.exceptions import VectorStoreError
from core.config.settings import settings
from .base import BaseVectorStore
from core.pipeline.utils.file_lock import file_lock as FileLock
from core.pipeline.embedders import create_embedder

logger = logging.getLogger(__name__)


def _l2_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@register("vectorstore", "faiss")
class FAISSVectorStore(BaseVectorStore):
    """
    Disk-persisted FAISS index (flat L2 or cosine).
    """

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        similarity: str = "cosine",
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.similarity = similarity

        self._index = None  # lazy
        self._metafile = None

    # ---------------- internal helpers ---------------- #
    def _load_index(self):
        if self._index is None and self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
        return self._index

    def _save_index(self):
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))

    def _append_metadata(self, rows: Sequence[Dict[str, Any]]):
        mode = "a" if self.metadata_path.exists() else "w"
        with open(self.metadata_path, mode, encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load_existing_ids(self, id_key: str = "id") -> set:
        """
        Load all existing IDs from the metadata file for deduplication.
        If id_key is not present, use a hash of the text.
        """
        existing_ids = set()
        if not self.metadata_path.exists():
            return existing_ids
        with open(self.metadata_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    meta = json.loads(line)
                    if id_key in meta:
                        existing_ids.add(meta[id_key])
                    else:
                        existing_ids.add(hash(meta.get("text", "")))
                except Exception:
                    continue
        return existing_ids

    # ---------------- impl: build / append ------------ #
    def _build_or_append(
        self,
        chunks: Sequence[Dict[str, Any]],
        embed_fn: Callable[[List[str]], List[List[float]]],
        batch_size: int,
        *,
        build: bool,
    ) -> None:
        if not chunks:
            return

        vecs = embed_fn([c["text"] for c in chunks])
        vecs = np.array(vecs).astype("float32")
        if self.similarity == "cosine":
            vecs = _l2_normalize(vecs)

        if build or not self.index_path.exists():
            dim = vecs.shape[1]
            if self.similarity == "l2":
                index = faiss.IndexFlatL2(dim)
            elif self.similarity in ("ip", "dot", "inner_product"):
                index = faiss.IndexFlatIP(dim)
            else:
                # defaulting to cosine if unspecified
                index = faiss.IndexFlatIP(dim)
            # new index created, so add vectors
            index.add(vecs)  # type: ignore[call-arg]
            self._index = index
            self._save_index()
        else:
            index = self._load_index()
            if index is None:
                raise VectorStoreError("No existing index to append to")
            index.add(vecs)  # type: ignore[call-arg]
            self._save_index()

        self._append_metadata(chunks)

    def build_index(
        self,
        chunks: Sequence[Dict[str, Any]],
        embed_fn: Callable[[List[str]], List[List[float]]],
        *,
        batch_size: int = 64,
    ) -> None:
        """
        Build a brand-new FAISS index for these chunks, overwriting any existing one.
        """
        lock_path = self.index_path.with_suffix(".lock")
        with FileLock(lock_path):
            self._build_or_append(chunks, embed_fn, batch_size, build=True)

    def append_to_index(
        self,
        chunks: Sequence[Dict[str, Any]],
        embed_fn: Callable[[List[str]], List[List[float]]],
        *,
        batch_size: int = 64,
    ) -> None:
        """
        Append vectors & metadata for these chunks to the existing index.
        """
        lock_path = self.index_path.with_suffix(".lock")
        with FileLock(lock_path):
            self._build_or_append(chunks, embed_fn, batch_size, build=False)

    def add(self, embeddings, metadatas, texts, id_key: str = "id"):
        """
        Add new vectors and metadata to the FAISS index, skipping duplicates by id_key or text hash.
        """
        if not embeddings:
            return
        existing_ids = self._load_existing_ids(id_key)
        new_vecs = []
        new_chunks = []
        for emb, meta, text in zip(embeddings, metadatas, texts):
            unique_id = meta.get(id_key) if id_key in meta else hash(text)
            if unique_id in existing_ids:
                continue
            chunk = dict(meta)
            chunk["text"] = text
            new_chunks.append(chunk)
            new_vecs.append(emb)
            existing_ids.add(unique_id)
        if not new_vecs:
            logger.info("No new unique entries to add to FAISS index.")
            return
        vecs = np.array(new_vecs).astype("float32")
        if self.similarity == "cosine":
            vecs = _l2_normalize(vecs)
        lock_path = self.index_path.with_suffix(".lock")
        with FileLock(lock_path):
            if not self.index_path.exists():
                dim = vecs.shape[1]
                if self.similarity == "l2":
                    index = faiss.IndexFlatL2(dim)
                elif self.similarity in ("ip", "dot", "inner_product"):
                    index = faiss.IndexFlatIP(dim)
                else:
                    index = faiss.IndexFlatIP(dim)
                index.add(vecs)
                self._index = index
                self._save_index()
            else:
                index = self._load_index()
                if index is None:
                    raise VectorStoreError("No existing index to append to")
                index.add(vecs)
                self._save_index()
            self._append_metadata(new_chunks)

    # ---------------- impl: retrieve ------------------ #
    def search(
        self,
        query: str,
        k: int,
    ) -> List[Document]:
        index = self._load_index()
        if index is None:
            raise VectorStoreError("Index not found")

        embedder_name = settings.get("EMBEDDER_PROVIDER", "openai")
        embedder = create_embedder(embedder_name)
        # Fix: Wrap the query in a list since embed_text expects a sequence of strings
        vec = np.array([embedder.embed_text([query])[0]]).astype("float32")
        if self.similarity == "cosine":
            vec = _l2_normalize(vec)

        # Pylance stub mismatch: ignore call-arg
        dists, idxs = index.search(vec, k)  # type: ignore[call-arg]
        docs: List[Document] = []
        with open(self.metadata_path, encoding="utf-8") as fh:
            lines = fh.readlines()

        for i in idxs[0]:
            if i < len(lines):
                meta = json.loads(lines[i])
                docs.append(
                    Document(page_content=meta["text"], metadata=meta)
                )
        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents and return them with their similarity scores.

        Parameters
        ----------
        query : str
            The query string to find similar documents for
        k : int
            The number of documents to return

        Returns
        -------
        List[tuple[Document, float]]
            List of (Document, score) tuples where score indicates the similarity
        """
        index = self._load_index()
        if index is None:
            raise VectorStoreError("Index not found")

        print(f"DEBUG VS: Using similarity algorithm: {self.similarity}")

        embedder_name = settings.get("EMBEDDER_PROVIDER", "openai")
        embedder = create_embedder(embedder_name)
        # Fix: Wrap the query in a list since embed_text expects a sequence of strings
        vec = np.array([embedder.embed_text([query])[0]]).astype("float32")
        if self.similarity == "cosine":
            vec = _l2_normalize(vec)

        # Search returns distances and indices
        print(f"DEBUG VS: Searching FAISS index for query: '{query}', k={k}")
        dists, idxs = index.search(vec, k)  # type: ignore[call-arg]

        # Both dists and idxs are 2D arrays where the first dimension is the query index
        # Since we only have one query, we get the first row
        distances = dists[0]
        indices = idxs[0]

        print(f"DEBUG VS: FAISS search returned {len(indices)} results")
        print(f"DEBUG VS: distances: {distances}")
        print(f"DEBUG VS: indices: {indices}")

        results = []

        # Read all metadata lines in one go
        with open(self.metadata_path, encoding="utf-8") as fh:
            lines = fh.readlines()
            print(f"DEBUG VS: Read {len(lines)} metadata lines")

        # Loop through the results and create (document, score) pairs
        for i, distance in zip(indices, distances):
            # Skip invalid indices
            if i < 0 or i >= len(lines):
                print(f"DEBUG VS: Skipping invalid index {i}")
                continue

            try:
                meta = json.loads(lines[i])
                doc = Document(page_content=meta["text"], metadata=meta)
                # For cosine similarity with IP index, translate IP distance to cosine distance
                distance_value = float(distance)

                # Create a proper tuple with exactly two elements
                result_tuple = (doc, distance_value)
                print(f"DEBUG VS: Created tuple with type: {type(result_tuple)}, doc: {type(doc)}, score: {distance_value}")
                results.append(result_tuple)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error processing document at index {i}: {e}")
                print(f"DEBUG VS: Error processing document at index {i}: {e}")
                continue

        print(f"DEBUG VS: Returning {len(results)} valid results")
        # Debug the final structure
        if results:
            first_result = results[0]
            print(f"DEBUG VS: First result type: {type(first_result)}")
            print(f"DEBUG VS: Is tuple? {isinstance(first_result, tuple)}")
            print(f"DEBUG VS: Tuple length: {len(first_result)}")
            print(f"DEBUG VS: First element type: {type(first_result[0])}")
            print(f"DEBUG VS: Second element type: {type(first_result[1])}")

        # Return a list of tuples, each containing (Document, float)
        return results

