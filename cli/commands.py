"""Slash-command and Typer CLI handlers for Coupa AI Assistant.

Provides an extendable command dispatcher for both interactive chat and CLI usage.
Defines Typer commands, slash-command registration, and bridges to core logic.
See docs/architecture/commands.md for design rationale and extension patterns.
"""

from __future__ import annotations

import logging
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, Any, Optional


import typer

from core.utils.i18n import get_message
from core.generation.rag_processor import process_rag_request, RAGRequest, RAGResponse
from core.config import settings
from core.utils.component_registry import register_cli_component
from core.llm import get_llm_client, get_embedder
from core.rag.classify import classify_question

# --------------------------------------------------------------------------- #
# Minimal in‑memory dispatcher for slash‑commands (when running in a TUI/Chat)
# --------------------------------------------------------------------------- #
class CommandDispatcher:
    """In-memory dispatcher for slash-commands in TUI/Chat.

    Allows registration and dispatch of string-based commands for chat interfaces.
    See docs/architecture/commands.md for design rationale.
    """
    def __init__(self):
        self._cmds: Dict[str, Callable[[str], str]] = {}

    def register(self, name: str):
        """Register a function as a slash-command."""

        def wrapper(fn):
            register_cli_component("command", name)(fn)
            self._cmds[name] = fn
            return fn

        return wrapper

    def dispatch(self, line: str) -> str | None:
        """Dispatch a line to the appropriate slash-command handler.

        Args:
            line (str): Input line starting with '/'.

        Returns:
            str | None: Command result or None if not a command.
        """
        if not line.startswith("/"):
            return None
        name, _, payload = line[1:].partition(" ")
        if name in self._cmds:
            return self._cmds[name](payload)
        return get_message("command.unknown", command=name)


_dispatcher = CommandDispatcher()

# --------------------------------------------------------------------------- #
# Typer app –entry‑point is exposed in pyproject.toml as "aiqa"              #
# --------------------------------------------------------------------------- #
app = typer.Typer(help=get_message("help.cli"))


@app.callback(invoke_without_command=False)
def _root_options(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "WARNING",
        "--log-level",
        help=get_message("help.log_level"),
        show_default=True,
        case_sensitive=False,
    ),
):
    """Shared option processed before any sub‑command executes."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format="%(levelname)s: %(message)s",
    )


# --------------------------------------------------------------------------- #
# Small decorator so one function registers as both Typer & slash‑command     #
# --------------------------------------------------------------------------- #
def command(name: str):
    """Decorator to register a function as both Typer and slash-command.

    Args:
        name (str): Command name.

    Returns:
        Callable: Decorator for the function.
    """
    def _decorator(fn: Callable[[str], str]):
        # slash‑command
        _dispatcher.register(name)(fn)

        # Typer sub‑command (payload passed as single arg)
        @app.command(name)
        def _typer_cmd(payload: str = typer.Argument("", help=get_message("param.command_payload"))):
            res = fn(payload)
            if res:
                typer.echo(res)

        return fn

    return _decorator


# --------------------------------------------------------------------------- #
# /ping — simple health check                                                 #
# --------------------------------------------------------------------------- #
@command("ping")
def _ping(_: str) -> str:
    return get_message("ping.response")


# --------------------------------------------------------------------------- #
# Export‑guides logic (used by Typer cmd & slash variant)                     #
# --------------------------------------------------------------------------- #
def export_guides_cli(
        exporter: str = settings.get("DEFAULT_EXPORTER", "cso_html"),
        workbook: Optional[str] = None,
        assets: Optional[str] = None,
        output: str = settings.get("HTML_EXPORT_DIR"),
        limit: int | None = settings.get("EXPORT_LIMIT", 0),
        no_captions: bool = settings.get("NO_CAPTIONS", False),
        force: bool = settings.get("FORCE_REGENERATE_DEFAULT", False),
):
    """CLI wrapper for exporting guides to HTML."""
    try:
        from core.services.export_service import export_guides

        # Call service function
        result = export_guides(
            exporter=exporter,
            workbook=workbook,
            assets=assets,
            output_dir=output,
            limit=limit if limit and limit > 0 else None,
            no_captions=no_captions,
            force=force
        )

        return result
    except ValueError as e:
        # Handle validation errors with specific messages
        typer.echo(get_message("command.failure", error=str(e)))
        raise typer.Exit(1)
    except Exception as e:
        # Handle unexpected errors
        typer.echo(get_message("command.failure", error=str(e)))
        raise typer.Exit(1)


@app.command("export-guides", help=get_message("export_guides.help"))
def _export_guides(
        exporter: str = typer.Option(
            settings.get("DEFAULT_EXPORTER", "cso_html"),
            help=get_message("export_guides.param.exporter_type")
        ),
        workbook: Optional[str] = typer.Option(
            None,
            help=get_message("export_guides.param.workbook")
        ),
        assets: Optional[str] = typer.Option(
            None,
            help=get_message("export_guides.param.assets")
        ),
        output: str = typer.Option(
            settings.get("HTML_EXPORT_DIR"),
            help=get_message("export_guides.param.output")
        ),
        limit: int = typer.Option(
            settings.get("EXPORT_LIMIT", 0),
            help=get_message("export_guides.param.limit")
        ),
        no_captions: bool = typer.Option(
            settings.get("NO_CAPTIONS", False),
            help=get_message("export_guides.param.no_captions")
        ),
        force: bool = typer.Option(
            settings.get("FORCE_REGENERATE_DEFAULT", False),
            "--force",
            help=get_message("param.force")
        ),
):
    """Export guides to HTML format command."""
    typer.echo(get_message("export_guides.starting"))
    typer.echo(f"Using exporter: {exporter}")
    if workbook:
        typer.echo(get_message("export_guides.workbook_info", workbook=workbook))
    if assets:
        typer.echo(get_message("export_guides.assets_info", assets=assets))
    typer.echo(get_message("export_guides.output_info", output=output))
    typer.echo(get_message("export_guides.limit_info", limit=limit or 'all'))

    res = export_guides_cli(exporter, workbook, assets, output, limit, no_captions, force)
    typer.echo(get_message("export_guides.complete", output=res))


# --------------------------------------------------------------------------- #
# Launch chat interface command                                               #
# --------------------------------------------------------------------------- #
@app.command("chat", help=get_message("chat.welcome"))
def _launch_chat():
    """Launch the interactive chat interface."""
    from cli.chat_cli_interface import run_chat_interface
    run_chat_interface()


# --------------------------------------------------------------------------- #
# Delegate component operations to the service layer                          #
# --------------------------------------------------------------------------- #
def run_component(component_type: str, component_name: str, args: Dict[Any, Any]) -> Any:
    """CLI wrapper for running components."""
    from core.services.component_service import run_component as run_component_service
    return run_component_service(component_type, component_name, args)


# Component subcommand
component_app = typer.Typer(help=get_message("help.component"))
app.add_typer(component_app, name="component")

# Chunker command
chunker_app = typer.Typer(help=get_message("help.chunker"))
component_app.add_typer(chunker_app, name="chunker")

@chunker_app.command("run", help=get_message("help.chunker_run"))
def run_chunker(
    name: str = typer.Argument(..., help=get_message("param.chunker_name")),
    input_file: Path = typer.Option(..., "--input", "-i", help=get_message("param.input_file")),
    output_file: Path = typer.Option(..., "--output", "-o", help=get_message("param.output_file")),
    chunk_size: int = typer.Option(
        None, "--chunk-size",
        help=get_message(
            "param.chunk_size",
            default_chunk_size=settings.get('DEFAULT_CHUNK_SIZE', 800)
        )
    ),
    chunk_overlap: int = typer.Option(
        None, "--chunk-overlap",
        help=get_message(
            "param.chunk_overlap",
            default_chunk_overlap=settings.get('DEFAULT_CHUNK_OVERLAP', 100)
        )
    ),
):
    """Run a document chunker."""
    try:
        # Read input file
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Prepare arguments
        args = {
            "content": content,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }

        # Run chunker
        result = run_component("chunker", name, args)

        # Write output
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        typer.echo(get_message(
            "component.chunker.processed",
            name=name,
            input_file=input_file,
            chunk_count=len(result)
        ))
        typer.echo(get_message("file.written", output_path=output_file))
    except Exception as e:
        typer.echo(get_message("command.failure", error=str(e)))
        raise typer.Exit(1)


# Cleaner command
cleaner_app = typer.Typer(help=get_message("help.cleaner"))
component_app.add_typer(cleaner_app, name="cleaner")

@cleaner_app.command("run", help=get_message("help.cleaner_run"))
def run_cleaner(
    name: str = typer.Argument(..., help=get_message("param.cleaner_name")),
    input_file: Path = typer.Option(..., "--input", "-i", help=get_message("param.input_file")),
    output_file: Path = typer.Option(..., "--output", "-o", help=get_message("param.output_file")),
):
    """Run a document cleaner."""
    try:
        # Read input file
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Run cleaner
        result = run_component("cleaner", name, {"content": content})

        # Write output
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)

        typer.echo(get_message(
            "component.cleaner.processed",
            name=name,
            input_file=input_file
        ))
        typer.echo(get_message("file.written", output_path=output_file))
    except Exception as e:
        typer.echo(get_message("command.failure", error=str(e)))
        raise typer.Exit(1)


# Exporter command
exporter_app = typer.Typer(help=get_message("help.exporter"))
component_app.add_typer(exporter_app, name="exporter")

@exporter_app.command("run", help=get_message("help.exporter_run"))
def run_exporter(
    name: str = typer.Argument(..., help=get_message("param.exporter_name")),
    output: Path = typer.Option(..., "--output", "-o", help=get_message("param.output_dir")),
    limit: Optional[int] = typer.Option(None, help=get_message("param.limit")),
    force: bool = typer.Option(False, help=get_message("param.force")),
    # Dataset-specific parameters
    workbook: Optional[Path] = typer.Option(None, help=get_message("dataset.cso.workbook_help")),
    assets: Optional[Path] = typer.Option(None, help=get_message("dataset.cso.assets_help")),
    parquet_file: Optional[Path] = typer.Option(None, help=get_message("dataset.parquet.file_help")),
):
    """Run a data exporter."""
    try:
        # Prepare exporter arguments
        exporter_args = {
            "out_dir": output,
            "limit": limit,
            "force_regenerate": force
        }

        # Add optional parameters if provided
        if workbook:
            exporter_args["workbook"] = workbook
        if assets:
            exporter_args["assets_dir"] = assets
        if parquet_file:
            exporter_args["parquet_file"] = parquet_file

        # Run exporter
        result = run_component("exporter", name, exporter_args)

        typer.echo(get_message(
            "component.exporter.processed",
            name=name,
            output_dir=output
        ))
        return result
    except Exception as e:
        typer.echo(get_message("command.failure", error=str(e)))
        raise typer.Exit(1)


# RAG query command
@app.command("query", help=get_message("help.query"))
def _query(
    question: str = typer.Argument(..., help=get_message("help.query_question")),
    raw: bool = typer.Option(False, "--raw", help=get_message("help.query_raw")),
    k: int = typer.Option(settings.get("TOP_K", 5), help=get_message("help.query_k")),
):
    """Query the knowledge base with RAG."""
    try:
        llm_client = get_llm_client()

        def embed_fn(texts: list[str]) -> list[list[float]]:
            return llm_client.get_embeddings(texts)

        classification_result = classify_question(question)
        primary_domain = "default"
        if classification_result and classification_result.get("domains"):
            primary_domain = classification_result["domains"][0][0]
        else:
            typer.echo(get_message("warning.classification_failed", query=question), err=True)

        rag_request_obj = RAGRequest(
            question=question,
            primary_domain=primary_domain,
            top_k=k
        )

        def query_embed_fn(query_text: str) -> list[float]:
            return llm_client.get_embeddings([query_text])[0]

        result: RAGResponse = process_rag_request(
            request=rag_request_obj,
            embed_fn=query_embed_fn
        )

        if raw:
            typer.echo(json.dumps(result.to_dict(), indent=2))
        else:
            response = result.get_answer_with_citations()
            typer.echo(response)
    except Exception as e:
        typer.echo(get_message("command.failure", error=str(e)))
        raise typer.Exit(1)


# Utility Commands
@app.command("file-size", help=get_message("file_size.help"))
def analyze_file_size(
    path: str = typer.Argument(..., help=get_message("file_size.param.path")),
    sort_by: str = typer.Option("size", help=get_message("file_size.param.sort_by")),
    limit: int = typer.Option(20, help=get_message("file_size.param.limit")),
):
    """Analyze file sizes in a directory or show size of a single file."""
    target = Path(path)

    if not target.exists():
        typer.echo(get_message("file_size.path_not_exist", path=path))
        raise typer.Exit(1)

    if target.is_file():
        size = target.stat().st_size
        typer.echo(f"File: {target.name}")
        typer.echo(f"Size: {get_human_readable_size(size)}")
    else:
        files = []
        for file_path in target.glob("**/*"):
            if file_path.is_file():
                files.append({
                    "path": file_path,
                    "name": file_path.name,
                    "extension": file_path.suffix,
                    "size": file_path.stat().st_size
                })

        # Sort files
        if sort_by == "size":
            files.sort(key=lambda x: x["size"], reverse=True)
        elif sort_by == "name":
            files.sort(key=lambda x: x["name"])
        elif sort_by == "extension":
            files.sort(key=lambda x: x["extension"])

        # Limit number of files
        files = files[:limit]

        # Print results
        typer.echo(get_message("file_size.directory_info", path=path))
        typer.echo(get_message("file_size.total_files", count=len(files)))
        typer.echo(get_message("file_size.largest_files"))

        for i, file in enumerate(files, start=1):
            rel_path = file["path"].relative_to(target)
            typer.echo(f"{i}. {rel_path} - {get_human_readable_size(file['size'])}")


def get_human_readable_size(size_bytes: int) -> str:
    """Convert size in bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


@app.command("clean-output")
def clean_output(
    directory: str = typer.Option(settings.get("HTML_EXPORT_DIR"), help="Directory to clean"),
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
):
    """Clean output directory by deleting all files and subdirectories."""
    target = Path(directory)

    if not target.exists():
        typer.echo(f"Directory {directory} does not exist.")
        return

    if not confirm:
        sure = typer.confirm(f"Are you sure you want to delete all files in {directory}?")
        if not sure:
            typer.echo("Operation cancelled.")
            return

    # Delete all files and subdirectories
    try:
        # Use shutil.rmtree for the directory, then recreate it
        shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Successfully cleaned {directory}")
    except Exception as e:
        typer.echo(f"Error cleaning directory: {str(e)}")
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# Slash‑command wrappers                                                      #
# --------------------------------------------------------------------------- #
@_dispatcher.register("build-index")
def _slash_build_index(payload: str) -> str:
    parts = payload.strip().split()
    if len(parts) != 2:
        return "Usage: /build-index <html_dir> <out_dir>"

    try:
        from core.services.indexing_service import build_index
        html_path = Path(parts[0])
        out_path = Path(parts[1])

        result = build_index(
            html_dir=html_path,
            output_dir=out_path
        )
        return f"Index built at {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@_dispatcher.register("query")
def _slash_query(payload: str) -> str:
    default_idx = "vector_store/vector_store_all"
    raw = False
    k = 5  # Default k value

    # Parse arguments
    parts = payload.split()
    i = 0
    while i < len(parts):
        if parts[i] == "--raw":
            raw = True
            parts.pop(i)
        elif parts[i] == "--k" and i + 1 < len(parts):
            try:
                k = int(parts[i + 1])
                parts.pop(i)
                parts.pop(i)
            except ValueError:
                return "Invalid value for --k, must be a number"
        else:
            i += 1

    payload = " ".join(parts)

    if "|" in payload:
        idx, q_text = map(str.strip, payload.split("|", 1))
        idx = idx or default_idx
    else:
        idx, q_text = default_idx, payload.strip()

    if not q_text:
        return "Usage: /query [--raw] [--k NUM] [index_dir |] QUESTION"

    try:
        llm_client = get_llm_client()

        def query_embed_fn_slash(query_text_slash: str) -> list[float]:
            return llm_client.get_embeddings([query_text_slash])[0]

        classification_result_slash = classify_question(q_text)
        primary_domain_slash = "default"
        if classification_result_slash and classification_result_slash.get("domains"):
            primary_domain_slash = classification_result_slash["domains"][0][0]

        rag_request_obj_slash = RAGRequest(
            question=q_text,
            primary_domain=primary_domain_slash,
            top_k=k
        )

        result_slash: RAGResponse = process_rag_request(
            request=rag_request_obj_slash,
            embed_fn=query_embed_fn_slash,
            override_index=idx if idx != default_idx else ""
        )

        if raw:
            return json.dumps(result_slash.to_dict(), indent=2)
        else:
            return result_slash.get_answer_with_citations()

    except Exception as e:
        return f"Error: {str(e)}"


@app.command("build-index", help=get_message("build_index.help"))
def build_index_cli(
        html_dir: str = typer.Argument(..., help=get_message("build_index.param.html_dir")),
        out: str = typer.Argument(..., help=get_message("build_index.param.out_dir")),
        chunk: str = typer.Option("header", help=get_message("build_index.param.chunk")),
        limit: int | None = typer.Option(0, help=get_message("build_index.param.limit")),
        batch: int = typer.Option(64, help=get_message("build_index.param.batch")),
):
    """CLI wrapper for building a FAISS index from exported HTML guides."""
    try:
        from core.services.indexing_service import build_index

        html_path = Path(html_dir)
        out_path = Path(out)

        if not html_path.exists():
            typer.echo(get_message("build_index.dir_not_found", dir_path=html_path))
            raise typer.Exit(1)

        html_files = list(html_path.glob("*.html"))
        file_count = len(html_files)
        typer.echo(get_message("build_index.found_files",
                               file_count=file_count,
                               html_dir=html_path,
                               limit=limit if limit else "all"))

        typer.echo(get_message("build_index.building", out_path=out_path))

        result = build_index(
            html_dir=html_path,
            output_dir=out_path,
            chunk_strategy=chunk,
            limit=limit if limit and limit > 0 else None,
            batch_size=batch
        )

        typer.echo(get_message("build_index.complete", result=result))
    except Exception as e:
        typer.echo(get_message("command.failure", error=str(e)))
        raise typer.Exit(1)

# Export commands for use by other modules
dispatch = _dispatcher.dispatch
__all__ = ["app", "dispatch", "command", "run_component"]

