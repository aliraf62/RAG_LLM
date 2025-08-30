"""
CLI commands for user authentication and session management.

Provides login, logout, and session status commands that integrate
with the core authentication system.
"""

from __future__ import annotations
from typing import Optional
import typer
import logging
from pathlib import Path
import json

from core.auth import (
    get_current_context,
    set_current_context,
    clear_current_context,
    AuthContext,
    authenticate_user,
    create_session,
    invalidate_session,
    get_session
)
from core.config.settings import apply_customer_settings
from core.utils.exceptions import AuthenticationError
from core.config.paths import customer_path

from cli.commands import CommandDispatcher

logger = logging.getLogger(__name__)

# Register auth commands with the CLI system
app = typer.Typer(name="auth", help="User authentication commands")

# Create auth command dispatcher for chat interface
dispatch = CommandDispatcher()

@app.command("login")
def login_command(
    username: str = typer.Option(..., "--username", "-u", help="Username to authenticate"),
    password: str = typer.Option(..., "--password", "-p", help="Password to verify", prompt=True, hide_input=True),
    customer_id: str = typer.Option(..., "--customer", "-c", help="Customer identifier")
) -> None:
    """
    Log in to the system with username and password.

    Creates a new session and sets the current authentication context.
    """
    try:
        # Authenticate the user
        user = authenticate_user(username, password, customer_id)

        # Create a new session
        session = create_session(user.id, customer_id)

        # Set the current context
        context = AuthContext(
            customer_id=customer_id,
            user_id=user.id,
            session_id=session.id
        )
        set_current_context(context)

        # Apply customer settings
        apply_customer_settings(customer_id)

        # Save session to disk
        _save_current_session(session.id)

        typer.echo(f"Logged in as {username} for customer {customer_id}")
        typer.echo(f"Session: {session.id}")

    except AuthenticationError as e:
        typer.echo(f"Authentication failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("logout")
def logout_command() -> None:
    """
    Log out of the current session.

    Invalidates the current session and clears the authentication context.
    """
    context = get_current_context()
    if not context.is_authenticated:
        typer.echo("Not logged in")
        return

    # Get session info before clearing
    user_id = context.user_id
    customer_id = context.customer_id
    session_id = context.session_id

    # Invalidate the session
    invalidated = invalidate_session(session_id)

    # Clear the context
    clear_current_context()

    # Remove saved session file
    _remove_saved_session()

    if invalidated:
        typer.echo(f"Logged out user {user_id} from customer {customer_id}")
    else:
        typer.echo("Failed to invalidate session")


@app.command("status")
def status_command() -> None:
    """
    Show current authentication status.

    Displays information about the currently authenticated user and session.
    """
    context = get_current_context()

    if not context.is_authenticated:
        typer.echo("Not logged in")
        return

    typer.echo(f"Logged in as user: {context.user_id}")
    typer.echo(f"Customer: {context.customer_id}")
    typer.echo(f"Session: {context.session_id}")

    # Check if session is still valid
    if context.session_id and not get_session(context.session_id):
        typer.echo("Warning: Session has expired or is invalid")


@app.command("restore")
def restore_session() -> None:
    """
    Restore the previous session if available.

    Loads the saved session from disk and sets the current authentication context.
    """
    session_file = Path.home() / ".ai_assistant_session"

    if not session_file.exists():
        typer.echo("No saved session found")
        return

    try:
        with open(session_file, "r") as f:
            data = json.load(f)

        session_id = data.get("session_id")
        if not session_id:
            typer.echo("Invalid session data")
            return

        # Verify the session is still valid
        session = get_session(session_id)
        if not session:
            typer.echo("Session has expired or is invalid")
            _remove_saved_session()
            return

        # Set the current context
        context = AuthContext(
            customer_id=session.customer_id,
            user_id=session.user_id,
            session_id=session.id
        )
        set_current_context(context)

        # Apply customer settings
        apply_customer_settings(session.customer_id)

        typer.echo(f"Restored session for user {session.user_id} (customer: {session.customer_id})")

    except Exception as e:
        typer.echo(f"Failed to restore session: {e}", err=True)


# Register chat slash commands
@dispatch.register("login")
def login_slash_command(args: str) -> str:
    """Log in to the system with username and password."""
    parts = args.strip().split()
    if len(parts) < 2:
        return "Usage: /login username customer_id [password]"

    username = parts[0]
    customer_id = parts[1]

    # If password is provided, use it; otherwise prompt
    if len(parts) > 2:
        password = parts[2]
    else:
        password = typer.prompt("Password", hide_input=True)

    try:
        user = authenticate_user(username, password, customer_id)
        session = create_session(user.id, customer_id)

        context = AuthContext(
            customer_id=customer_id,
            user_id=user.id,
            session_id=session.id
        )
        set_current_context(context)

        # Apply customer settings
        apply_customer_settings(customer_id)

        # Save session to disk
        _save_current_session(session.id)

        return f"✅ Logged in as {username} for customer {customer_id}"

    except AuthenticationError as e:
        return f"❌ Authentication failed: {e}"


@dispatch.register("logout")
def logout_slash_command(args: str) -> str:
    """Log out of the current session."""
    context = get_current_context()
    if not context.is_authenticated:
        return "Not logged in"

    session_id = context.session_id
    invalidated = invalidate_session(session_id)
    clear_current_context()
    _remove_saved_session()

    if invalidated:
        return "✅ Logged out successfully"
    else:
        return "❌ Failed to invalidate session"


@dispatch.register("status")
def status_slash_command(args: str) -> str:
    """Show current authentication status."""
    context = get_current_context()

    if not context.is_authenticated:
        return "Not logged in"

    result = [
        f"👤 User: {context.user_id}",
        f"🏢 Customer: {context.customer_id}",
        f"🔑 Session: {context.session_id}"
    ]

    # Check if session is still valid
    if context.session_id and not get_session(context.session_id):
        result.append("⚠️ Warning: Session has expired or is invalid")

    return "\n".join(result)


# Helper functions
def _save_current_session(session_id: str) -> None:
    """Save the current session ID to a file for later restoration."""
    session_file = Path.home() / ".ai_assistant_session"

    with open(session_file, "w") as f:
        json.dump({"session_id": session_id}, f)


def _remove_saved_session() -> None:
    """Remove the saved session file."""
    session_file = Path.home() / ".ai_assistant_session"
    if session_file.exists():
        session_file.unlink()
