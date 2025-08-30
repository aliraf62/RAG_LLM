"""
Entry point for running the CLI as a module: `python -m cli`

This provides a consistent interface to both:
- Command-line functionality (ping, query, build-index, etc.)
- Interactive chat interface

Examples:
  python -m cli           # Launches chat interface by default
  python -m cli chat      # Explicitly launches chat interface
  python -m cli cli       # Accesses original CLI commands
  python -m cli ping      # Direct access to a specific command

This file delegates to the main application defined in the project root.
"""

import sys
import os

# Add parent directory to path to import from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main application entry point
from main import app

if __name__ == "__main__":
    app()