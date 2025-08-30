"""
Enhanced logging utilities for AI QnA Assistant.

This module provides improved logging and visualization functions that can be
used throughout the codebase to provide better insights into processing stages.
"""
from __future__ import annotations

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()

def log_stage(name: str):
    """Decorator to log the start/end of processing stages with timing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            console.print(f"\n[bold blue]Starting {name}...[/bold blue]")
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            # Format result summary
            if isinstance(result, list) and result:
                count = len(result)
                sample = str(result[0])[:100] + "..." if len(str(result[0])) > 100 else str(result[0])
                console.print(Panel(f"[green]Completed {name} in {elapsed:.2f}s[/green]\n"
                             f"[yellow]Items processed:[/yellow] {count}\n"
                             f"[yellow]Sample:[/yellow] {sample}"))
            else:
                console.print(f"[green]Completed {name} in {elapsed:.2f}s[/green]")
            return result
        return wrapper
    return decorator

def show_document_sample(docs: List[Any], count: int = 3):
    """Display a sample of documents in a formatted table."""
    table = Table(title=f"Document Sample (showing {min(count, len(docs))} of {len(docs)})")

    # Determine columns based on first document
    if not docs:
        console.print("[yellow]No documents to display[/yellow]")
        return

    sample_doc = docs[0]
    if hasattr(sample_doc, "page_content") and hasattr(sample_doc, "metadata"):
        # LangChain-style Document
        table.add_column("Content (first 100 chars)", style="cyan")
        table.add_column("Metadata", style="green")

        for doc in docs[:count]:
            content_preview = str(doc.page_content)[:100] + "..." if len(str(doc.page_content)) > 100 else str(doc.page_content)
            metadata_preview = str(doc.metadata)[:100] + "..." if len(str(doc.metadata)) > 100 else str(doc.metadata)
            table.add_row(content_preview, metadata_preview)
    elif isinstance(sample_doc, dict):
        # Dictionary-style document
        if "text" in sample_doc:
            table.add_column("Text (first 100 chars)", style="cyan")
            if "metadata" in sample_doc:
                table.add_column("Metadata", style="green")

            for doc in docs[:count]:
                content_preview = str(doc.get("text", ""))[:100] + "..." if len(str(doc.get("text", ""))) > 100 else str(doc.get("text", ""))
                row = [content_preview]
                if "metadata" in sample_doc:
                    metadata_preview = str(doc.get("metadata", ""))[:100] + "..." if len(str(doc.get("metadata", ""))) > 100 else str(doc.get("metadata", ""))
                    row.append(metadata_preview)
                table.add_row(*row)
        else:
            # Generic dictionary
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")

            for doc in docs[:count]:
                for k, v in doc.items():
                    table.add_row(k, str(v)[:100] + "..." if len(str(v)) > 100 else str(v))

    console.print(table)

def create_progress_bar(total: int, description: str) -> Progress:
    """Create a rich progress bar for tracking operations."""
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TextColumn("[yellow]{task.speed:.2f} items/s"),
        TextColumn("[green]ETA: {task.eta_string}"),
    )
    progress.add_task(description, total=total)
    return progress

def timed_operation(func: Callable) -> Callable:
    """Decorator to time any operation and print results."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        console.print(f"[bold cyan]{func.__name__}[/bold cyan] completed in [green]{elapsed:.2f}s[/green]")
        return result
    return wrapper
