"""
Vector-store indexing orchestrator — backend agnostic.
"""
from __future__ import annotations

import concurrent.futures
import inspect
import logging
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence

from langchain.schema import Document
from tqdm import tqdm
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.console import Console
from rich.panel import Panel
import threading
import pickle

from core.utils.component_registry import get as get_component
from core.config.paths import VECTOR_STORE_DIR
from core.config.settings import settings
from core.pipeline.embedders import get_embedder
from core.pipeline.base import Row

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _ensure(value: int | None, fallback: int) -> int:
    return fallback if value is None else int(value)


def chunk_iter(items: Iterable, n: int) -> Iterator[List]:
    it = iter(items)
    while chunk := list(islice(it, n)):
        yield chunk


def _provider(category: str, default: str):
    return get_component(category, settings.get(f"{category}_provider", default))


# --------------------------------------------------------------------------- #
# Vector-store factory                                                        #
# --------------------------------------------------------------------------- #
def _make_store(out_dir: Path):
    vs_cls = _provider("vectorstore", "faiss")

    ctor_kwargs: Dict[str, Any] = {}
    sig = inspect.signature(vs_cls)

    if "index_path" in sig.parameters:
        ctor_kwargs["index_path"] = out_dir / "index"
    if "metadata_path" in sig.parameters:
        ctor_kwargs["metadata_path"] = out_dir / "metadata.jsonl"
    if "similarity" in sig.parameters:
        ctor_kwargs["similarity"] = settings.get(
            "VECTORSTORE_SIMILARITY_ALGORITHM", "cosine"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    return vs_cls(**ctor_kwargs)  # type: ignore[arg-type]


def _default_embed_fn():
    emb_name = settings.get("embedder_provider", "openai")
    return get_embedder(emb_name)().embed_text


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def build_index_from_documents(
    *,
    docs: Sequence[Document],
    customer_id: str,
    dataset_id: str,
    embed_fn: Callable[[List[str]], List[List[float]]] | None = None,
    batch_size: int | None = None,
    worker_count: int | None = None,
    chunk_size: int | None = None,
    checkpoint_path: str | None = None,
    verbose: bool = False,
    max_retries: int = 3,
) -> Path:
    """
    Build / append a vector index for *customer_id / dataset_id*.
    Adds checkpointing, rich progress, and detailed logging.
    batch_size, worker_count, and chunk_size default to settings values if not provided.
    Config mapping:
      - batch_size: VECTORSTORE_BATCH_SIZE
      - worker_count: VECTORSTORE_WORKERS
      - chunk_size: VECTORSTORE_CHUNK_SIZE
    """
    batch_size = _ensure(batch_size, settings.get("VECTORSTORE_BATCH_SIZE"))
    worker_count = _ensure(worker_count, settings.get("VECTORSTORE_WORKERS"))
    chunk_size = _ensure(chunk_size, settings.get("VECTORSTORE_CHUNK_SIZE"))
    embed_fn = embed_fn or _default_embed_fn()

    # Resolve output dir respecting VECTORSTORE_SCOPE
    scope = settings.get("VECTORSTORE_SCOPE", "dataset")
    root = Path(settings.get("VECTORSTORE_OUTPUT_ROOT", VECTOR_STORE_DIR))
    if scope == "customer":
        out_dir = root / customer_id / "all"
    else:
        out_dir = root / customer_id / dataset_id
    store = _make_store(out_dir)

    # Checkpointing setup
    if checkpoint_path is None:
        checkpoint_path = str(out_dir / "checkpoint.pkl")
    processed_ids = set()
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            processed_ids = pickle.load(f)

    # Skip if present and regenerate flag is off
    if (
        out_dir.exists()
        and not settings.get("FORCE_REGENERATE_VECTOR", False)
        and (out_dir / "index").exists()
    ):
        logger.info("Vector store already exists. Skipping rebuild.")
        return out_dir

    # Filter docs to skip already processed
    def doc_id(doc):
        return doc.metadata.get("id") or doc.metadata.get("source") or hash(doc.page_content)
    docs_to_process = [d for d in docs if doc_id(d) not in processed_ids]
    if not docs_to_process:
        logger.info("All documents already processed. Nothing to do.")
        return out_dir

    # Split into batches
    doc_batches = list(chunk_iter(docs_to_process, chunk_size))
    total_batches = len(doc_batches)
    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=not verbose,
    )
    log_panel = []
    lock = threading.Lock()

    def log_event(msg):
        if verbose:
            console.log(msg)
        else:
            with lock:
                log_panel.append(msg)

    def save_checkpoint():
        with open(checkpoint_path, "wb") as f:
            pickle.dump(processed_ids, f)

    def process_batch(batch, batch_idx):
        ids = [doc_id(d) for d in batch]
        for attempt in range(1, max_retries+1):
            try:
                log_event(f"Worker {threading.get_ident()} processing batch {batch_idx+1}/{total_batches} (attempt {attempt})")
                store.append_to_index(batch, embed_fn, batch_size)
                with lock:
                    processed_ids.update(ids)
                    save_checkpoint()
                log_event(f"Batch {batch_idx+1} processed successfully.")
                return True
            except Exception as e:
                log_event(f"Batch {batch_idx+1} failed on attempt {attempt}: {e}")
                if attempt == max_retries:
                    log_event(f"Batch {batch_idx+1} permanently failed after {max_retries} attempts.")
                    return False

    # Single-thread path
    if worker_count <= 1:
        task = progress.add_task("Indexing", total=len(doc_batches))
        with progress:
            for i, batch in enumerate(doc_batches):
                process_batch(batch, i)
                progress.update(task, advance=1)
        return out_dir

    # Multi-threaded with progress bars
    with progress:
        task = progress.add_task("Indexing", total=len(doc_batches))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as ex:
            futs = {ex.submit(process_batch, b, i): i for i, b in enumerate(doc_batches)}
            for fut in concurrent.futures.as_completed(futs):
                progress.update(task, advance=1)
    if not verbose:
        # Show summary panel at the end
        console.print(Panel("\n".join(log_panel), title="Indexing Log", subtitle="(use verbose for live logs)"))
    return out_dir


def build_index_from_rows(
    *,
    rows: Sequence[Row],
    customer_id: str,
    dataset_id: str,
    embed_fn: Callable[[List[str]], List[List[float]]] | None = None,
    batch_size: int | None = None,
    worker_count: int | None = None,
    chunk_size: int | None = None,
    checkpoint_path: str | None = None,
    verbose: bool = False,
    max_retries: int = 3,
) -> Path:
    """
    Build / append a vector index for *customer_id / dataset_id* from Row objects.
    Adds checkpointing, rich progress, and detailed logging.
    """
    batch_size = _ensure(batch_size, settings.get("VECTORSTORE_BATCH_SIZE"))
    worker_count = _ensure(worker_count, settings.get("VECTORSTORE_WORKERS"))
    chunk_size = _ensure(chunk_size, settings.get("VECTORSTORE_CHUNK_SIZE"))
    embed_fn = embed_fn or _default_embed_fn()

    # Resolve output dir respecting VECTORSTORE_SCOPE
    scope = settings.get("VECTORSTORE_SCOPE", "dataset")
    root = Path(settings.get("VECTORSTORE_OUTPUT_ROOT", VECTOR_STORE_DIR))
    if scope == "customer":
        out_dir = root / customer_id / "all"
    else:
        out_dir = root / customer_id / dataset_id
    store = _make_store(out_dir)

    # Checkpointing setup
    if checkpoint_path is None:
        checkpoint_path = str(out_dir / "checkpoint.pkl")
    processed_ids = set()
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as f:
            processed_ids = pickle.load(f)

    def row_id(row: Row):
        return row.id or row.metadata.get("id") or row.metadata.get("source") or hash(row.text)
    rows_to_process = [r for r in rows if row_id(r) not in processed_ids]
    if not rows_to_process:
        logger.info("All rows already processed. Nothing to do.")
        return out_dir

    row_batches = list(chunk_iter(rows_to_process, chunk_size))
    total_batches = len(row_batches)
    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=not verbose,
    )
    log_panel = []
    lock = threading.Lock()

    def log_event(msg):
        if verbose:
            console.log(msg)
        else:
            with lock:
                log_panel.append(msg)

    def save_checkpoint():
        with open(checkpoint_path, "wb") as f:
            pickle.dump(processed_ids, f)

    def process_batch(batch, batch_idx):
        ids = [row_id(r) for r in batch]
        for attempt in range(1, max_retries+1):
            try:
                log_event(f"Worker {threading.get_ident()} processing batch {batch_idx+1}/{total_batches} (attempt {attempt})")
                store.append_to_index_from_rows(batch, embed_fn, batch_size)
                with lock:
                    processed_ids.update(ids)
                    save_checkpoint()
                log_event(f"Batch {batch_idx+1} processed successfully.")
                return True
            except Exception as e:
                log_event(f"Batch {batch_idx+1} failed on attempt {attempt}: {e}")
                if attempt == max_retries:
                    log_event(f"Batch {batch_idx+1} permanently failed after {max_retries} attempts.")
                    return False

    if worker_count <= 1:
        task = progress.add_task("Indexing", total=len(row_batches))
        with progress:
            for i, batch in enumerate(row_batches):
                process_batch(batch, i)
                progress.update(task, advance=1)
        return out_dir

    with progress:
        task = progress.add_task("Indexing", total=len(row_batches))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as ex:
            futs = {ex.submit(process_batch, b, i): i for i, b in enumerate(row_batches)}
            for fut in concurrent.futures.as_completed(futs):
                progress.update(task, advance=1)
    if not verbose:
        console.print(Panel("\n".join(log_panel), title="Indexing Log", subtitle="(use verbose for live logs)"))
    return out_dir