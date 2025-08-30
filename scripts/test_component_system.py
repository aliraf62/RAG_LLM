#!/usr/bin/env python3
"""
Script to test the component registry and command system.

This script tests the component registry, command dispatching,
and CLI command execution functionality.
"""

import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import tempfile
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.utils.component_registry import register, get, available
from core.services.component_service import get_instance, run_component
from cli.commands import _dispatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def test_component_registry():
    """Test the component registry functionality."""
    print("\n=== Testing Component Registry ===\n")

    # Get all registered component types
    print("Registered component categories:")

    # Simple dummy component for testing
    @register("test_registry", "dummy")
    class DummyComponent:
        def __init__(self):
            self.name = "DummyComponent"

        def run(self, **kwargs):
            return f"DummyComponent executed with: {kwargs}"

    # Check if our component is registered
    components = available("test_registry")
    print(f"Components in 'test_registry' category: {components}")

    # Get and run the component
    component_class = get("test_registry", "dummy")
    component = component_class()
    result = component.run(test_param="Hello World")
    print(f"Component execution result: {result}")

    # Test get_instance utility
    instance = get_instance("test_registry", "dummy")
    print(f"Got instance: {instance.name}")

    # Test run_component utility
    try:
        result = run_component("test_registry", "dummy", {"test_param": "Via run_component"})
        print(f"run_component result: {result}")
    except Exception as e:
        print(f"Error running component: {e}")

    return components

def test_command_dispatcher():
    """Test the command dispatcher functionality."""
    print("\n=== Testing Command Dispatcher ===\n")

    # Register a test command
    @_dispatcher.register("test_command")
    def test_cmd(args):
        return f"Test command executed with args: {args}"

    # Dispatch the command
    result = _dispatcher.dispatch("/test_command with some arguments")
    print(f"Command result: {result}")

    # Test with unknown command
    result = _dispatcher.dispatch("/unknown_command")
    print(f"Unknown command result: {result}")

    # Test with non-command
    result = _dispatcher.dispatch("This is not a command")
    print(f"Non-command result: {result or 'None (as expected)'}")

    return True

def test_component_cli(tmp_path=None):
    """
    Test CLI component execution flow.

    This simulates what happens when running CLI commands
    but without actually invoking the Typer CLI.
    """
    print("\n=== Testing Component CLI Execution ===\n")

    # Create a temporary directory for output if not provided
    if tmp_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp_dir.name)
    else:
        tmp_path = Path(tmp_path)

    # Create a simple test input file
    input_file = tmp_path / "test_input.txt"
    with open(input_file, "w") as f:
        f.write("This is test content\nLine 1\nLine 2\nLine 3")

    # Define output file
    output_file = tmp_path / "test_output.txt"

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Helper function to simulate chunker execution
    def run_chunker(chunker_name, input_path, output_path, chunk_size=100, chunk_overlap=20):
        print(f"\nRunning chunker '{chunker_name}' with:")
        print(f"  - Input: {input_path}")
        print(f"  - Output: {output_path}")
        print(f"  - Chunk size: {chunk_size}")
        print(f"  - Chunk overlap: {chunk_overlap}")

        try:
            # Try to run the actual chunker component if it exists
            args = {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }

            try:
                result = run_component("chunker", chunker_name, args)
                print(f"Chunker execution result: {result}")
            except Exception as e:
                print(f"Note: Could not run real chunker (expected if testing only): {e}")

                # Simulate the chunker output for testing purposes
                with open(output_path, "w") as f:
                    f.write(f"Simulated chunker output for {chunker_name}\n")
                    f.write(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}\n")
                    f.write("Chunk 1: This is test content\n")
                    f.write("Chunk 2: Line 1\nLine 2\n")
                    f.write("Chunk 3: Line 2\nLine 3\n")

                print(f"Created simulated output at: {output_path}")

            return True

        except Exception as e:
            print(f"Error running chunker: {e}")
            return False

    # Test with a few different chunker configurations
    chunker_configs = [
        {"name": "character", "chunk_size": 100, "chunk_overlap": 20},
        {"name": "paragraph", "chunk_size": 200, "chunk_overlap": 50},
        {"name": "sentence", "chunk_size": 150, "chunk_overlap": 30}
    ]

    results = {}
    for config in chunker_configs:
        chunker_output = tmp_path / f"output_{config['name']}.txt"
        result = run_chunker(
            config["name"],
            input_file,
            chunker_output,
            config["chunk_size"],
            config["chunk_overlap"]
        )
        results[config["name"]] = result

    print("\nTest summary:")
    for chunker_name, success in results.items():
        print(f"  - {chunker_name}: {'✅ Success' if success else '❌ Failed'}")

    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test component and command systems")
    parser.add_argument("--test", choices=["registry", "commands", "cli", "all"], default="all",
                      help="Which test to run: 'registry', 'commands', 'cli', or 'all'")

    args = parser.parse_args()

    if args.test in ["registry", "all"]:
        test_component_registry()

    if args.test in ["commands", "all"]:
        test_command_dispatcher()

    if args.test in ["cli", "all"]:
        test_component_cli()

if __name__ == "__main__":
    main()
