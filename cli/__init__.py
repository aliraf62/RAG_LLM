"""
Allow the package to be executed with `python -m cli ...`.

It simply forwards execution to the Typer application defined in
`cli.commands`.
"""

from .commands import app


def main() -> None:
    """Entry point for CLI package.

    Runs the Typer application defined in cli.commands.
    """
    app()


if __name__ == "__main__":
    main()