# cli/component_cli.py
"""Standalone command-line interface for data ingestion components.

This module provides a separate entry point for running data ingestion components
directly from the command line, without using the main 'aiqa' command. It reuses
the core implementation from cli.commands to avoid duplication.
"""

import argparse
import sys
import json
from pathlib import Path

from core.config.settings import settings
from cli.commands import run_component
from core.utils.i18n import get_message
from core.utils.component_registry import component_message


def setup_chunker_parser(subparsers, component_registry):
    """Set up parser for chunker components."""
    chunker_parser = subparsers.add_parser("chunker", help=get_message("help.chunker"))
    chunker_parser.add_argument(
        "name",
        choices=component_registry["chunker"].keys(),
        help=get_message("param.chunker_name")
    )
    chunker_parser.add_argument(
        "--input", "-i",
        required=True,
        help=get_message("param.input_file")
    )
    chunker_parser.add_argument(
        "--output", "-o",
        required=True,
        help=get_message("param.output_file")
    )
    chunker_parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.get("DEFAULT_CHUNK_SIZE"),
        help=get_message("param.chunk_size", default_chunk_size=settings.get("DEFAULT_CHUNK_SIZE"))
    )
    chunker_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.get("DEFAULT_CHUNK_OVERLAP"),
        help=get_message("param.chunk_overlap", default_chunk_overlap=settings.get("DEFAULT_CHUNK_OVERLAP"))
    )


def setup_cleaner_parser(subparsers, component_registry):
    """Set up parser for cleaner components."""
    cleaner_parser = subparsers.add_parser("cleaner", help=get_message("help.cleaner"))
    cleaner_parser.add_argument(
        "name",
        choices=component_registry["cleaner"].keys(),
        help=get_message("param.cleaner_name")
    )
    cleaner_parser.add_argument(
        "--input", "-i",
        required=True,
        help=get_message("param.input_file")
    )
    cleaner_parser.add_argument(
        "--output", "-o",
        required=True,
        help=get_message("param.output_file")
    )


def setup_exporter_parser(subparsers, component_registry):
    """Set up parser for exporter components."""
    exporter_parser = subparsers.add_parser("exporter", help=get_message("help.exporter"))
    exporter_parser.add_argument(
        "--source", "-s",
        help="Source file or directory"
    )
    exporter_parser.add_argument(
        "--source-type",
        choices=["excel", "parquet", "text", "pdf", "auto"],
        default="auto",
        help="Source type (auto-detected by default)"
    )
    exporter_parser.add_argument(
        "name",
        nargs="?",  # Make optional when source is specified
        choices=component_registry["exporter"].keys(),
        help=get_message("param.exporter_name")
    )
    exporter_parser.add_argument(
        "--output", "-o",
        required=True,
        help=get_message("param.output_dir")
    )
    exporter_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=get_message("param.limit")
    )
    exporter_parser.add_argument(
        "--force",
        action="store_true",
        help=get_message("param.force")
    )

    # Format-specific options but generalized
    exporter_parser.add_argument(
        "--workbook",
        help="Path to the Excel workbook file"
    )
    exporter_parser.add_argument(
        "--assets",
        help="Path to the assets directory"
    )
    exporter_parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Disable image captioning"
    )


def main():
    """Main entry point for component CLI."""
    component_registry = settings.get("CLI_COMPONENTS", {})

    parser = argparse.ArgumentParser(description=get_message("help.component"))
    subparsers = parser.add_subparsers(dest="component_type", help=get_message("help.component"))

    # Create subparsers for each component type
    if "chunker" in component_registry:
        setup_chunker_parser(subparsers, component_registry)

    if "cleaner" in component_registry:
        setup_cleaner_parser(subparsers, component_registry)

    if "exporter" in component_registry:
        setup_exporter_parser(subparsers, component_registry)

    # Parse arguments
    args = parser.parse_args()

    if not args.component_type:
        parser.print_help()
        return

    try:
        if args.component_type == "chunker":
            # Read input file
            with open(args.input, "r", encoding="utf-8") as f:
                content = f.read()

            # Prepare arguments for chunker
            component_args = {
                "content": content,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap
            }

            # Run chunker
            result = run_component(args.component_type, args.name, component_args)

            # Write result to output file
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(get_message("component.chunker.processed",
                              name=args.name,
                              input_file=args.input,
                              chunk_count=len(result)
                              ))
            print(get_message("file.written", output_path=args.output))

        elif args.component_type == "cleaner":
            # Read input file
            with open(args.input, "r", encoding="utf-8") as f:
                content = f.read()

            # Run cleaner
            result = run_component(args.component_type, args.name, {"content": content})

            # Write result to output file
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result)

            print(get_message("component.cleaner.processed",
                              name=args.name,
                              input_file=args.input
                              ))
            print(get_message("file.written", output_path=args.output))

        elif args.component_type == "exporter":
            # Prepare arguments for exporter
            component_args = {
                "out_dir": Path(args.output),
                "limit": args.limit,
                "force_regenerate": args.force
            }

            # Add optional parameters if provided
            if hasattr(args, 'workbook') and args.workbook:
                component_args["workbook"] = Path(args.workbook)

            if hasattr(args, 'assets') and args.assets:
                component_args["assets_dir"] = Path(args.assets)

            if hasattr(args, 'parquet_file') and args.parquet_file:
                component_args["parquet_file"] = Path(args.parquet_file)

            # Run exporter
            result = run_component(args.component_type, args.name, component_args)

            print(get_message("component.exporter.processed",
                              name=args.name,
                              output_dir=args.output
                              ))
            print(get_message("command.success",
                              component_type=args.component_type,
                              component_name=args.name
                              ))

    except Exception as e:
        print(get_message("command.failure", error=str(e)))
        sys.exit(1)


if __name__ == "__main__":
    main()

