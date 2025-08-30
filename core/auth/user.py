"""
core.auth.user
==============

User models and authentication logic.

Provides user management functionality with customer-specific user roles.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import logging
import hashlib
import os
import json
from pathlib import Path
import time

from core.config.paths import project_path, customer_path
from core.utils.models import SingletonMeta
from core.utils.exceptions import AuthenticationError

logger = logging.getLogger(__name__)


@dataclass
class User:
    """
    User model representing an authenticated user in the system.

    Each user belongs to a specific customer and has a set of roles
    that determine their permissions.
    """

    id: str
    username: str
    email: str
    customer_id: str
    full_name: Optional[str] = None
    roles: Set[str] = field(default_factory=set)
    metadata: Dict[str, str] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        """Check if the user has a specific role."""
        return role in self.roles

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"User(id={self.id!r}, username={self.username!r}, customer_id={self.customer_id!r})"

    @classmethod
    def from_dict(cls, data: Dict) -> User:
        """Create a User instance from a dictionary."""
        roles = set(data.pop("roles", []))
        return cls(**data, roles=roles)


class UserStore(metaclass=SingletonMeta):
    """
    Store and retrieve user information.

    This class manages user data persistence and retrieval.
    For simplicity, users are stored in JSON files per customer.
    In a production environment, this would use a database.
    """

    def __init__(self) -> None:
        """Initialize the user store."""
        self._users_by_id: Dict[str, User] = {}
        self._users_by_username: Dict[str, User] = {}
        self._credentials: Dict[str, Dict[str, str]] = {}  # customer -> {username -> password_hash}
        self._loaded_customers: Set[str] = set()

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by their ID."""
        return self._users_by_id.get(user_id)

    def get_user_by_username(self, username: str, customer_id: str) -> Optional[User]:
        """Get a user by their username and customer ID."""
        self._ensure_customer_loaded(customer_id)
        key = f"{customer_id}:{username}"
        return self._users_by_username.get(key)

    def authenticate(self, username: str, password: str, customer_id: str) -> Optional[User]:
        """
        Authenticate a user by username and password.

        Parameters
        ----------
        username : str
            Username to authenticate
        password : str
            Password to verify
        customer_id : str
            Customer identifier

        Returns
        -------
        Optional[User]
            Authenticated user or None if authentication failed

        Raises
        ------
        AuthenticationError
            If authentication fails
        """
        self._ensure_customer_loaded(customer_id)

        # Check if the customer and username exist
        key = f"{customer_id}:{username}"
        user = self._users_by_username.get(key)
        if not user:
            logger.warning(f"Authentication failed: User {username} not found for customer {customer_id}")
            raise AuthenticationError(f"User {username} not found for customer {customer_id}")

        # Check if the password matches
        customer_creds = self._credentials.get(customer_id, {})
        stored_hash = customer_creds.get(username)
        if not stored_hash:
            logger.error(f"Credentials missing for user {username} of customer {customer_id}")
            raise AuthenticationError(f"Credentials missing for user {username}")

        # Verify password
        if self._hash_password(password) != stored_hash:
            logger.warning(f"Authentication failed: Invalid password for user {username}")
            raise AuthenticationError(f"Invalid password for user {username}")

        return user

    def _ensure_customer_loaded(self, customer_id: str) -> None:
        """Ensure a customer's users are loaded from disk."""
        if customer_id in self._loaded_customers:
            return

        try:
            self._load_users_for_customer(customer_id)
            self._loaded_customers.add(customer_id)
        except Exception as e:
            logger.error(f"Failed to load users for customer {customer_id}: {e}")

    def _load_users_for_customer(self, customer_id: str) -> None:
        """Load users from disk for a specific customer."""
        # Get the customer's root directory
        customer_root = customer_path(customer_id)
        users_file = customer_root / "config" / "users.json"
        credentials_file = customer_root / "config" / "credentials.json"

        # Load users
        if users_file.exists():
            try:
                with open(users_file, "r") as f:
                    users_data = json.load(f)

                for user_dict in users_data:
                    user = User.from_dict(user_dict)
                    self._users_by_id[user.id] = user
                    key = f"{user.customer_id}:{user.username}"
                    self._users_by_username[key] = user

                logger.info(f"Loaded {len(users_data)} users for customer {customer_id}")
            except Exception as e:
                logger.error(f"Error loading users for customer {customer_id}: {e}")
        else:
            logger.warning(f"No users file found for customer {customer_id} at {users_file}")

        # Load credentials
        if credentials_file.exists():
            try:
                with open(credentials_file, "r") as f:
                    self._credentials[customer_id] = json.load(f)
                logger.info(f"Loaded credentials for customer {customer_id}")
            except Exception as e:
                logger.error(f"Error loading credentials for customer {customer_id}: {e}")
        else:
            logger.warning(f"No credentials file found for customer {customer_id} at {credentials_file}")

    def _hash_password(self, password: str) -> str:
        """
        Create a simple hash of a password.

        For production use, this should be replaced with a secure password
        hashing algorithm like bcrypt, Argon2, or PBKDF2.
        """
        # Simple SHA-256 hash for demonstration purposes
        # In production, use a proper password hashing library with salt
        return hashlib.sha256(password.encode()).hexdigest()


# Singleton instance
_user_store = UserStore()


def get_user(user_id: str) -> Optional[User]:
    """
    Get a user by their ID.

    Parameters
    ----------
    user_id : str
        User identifier

    Returns
    -------
    Optional[User]
        User instance or None if not found
    """
    return _user_store.get_user_by_id(user_id)


def authenticate_user(username: str, password: str, customer_id: str) -> User:
    """
    Authenticate a user with username and password.

    Parameters
    ----------
    username : str
        Username to authenticate
    password : str
        Password to verify
    customer_id : str
        Customer identifier

    Returns
    -------
    User
        Authenticated user

    Raises
    ------
    AuthenticationError
        If authentication fails
    """
    user = _user_store.authenticate(username, password, customer_id)
    if not user:
        raise AuthenticationError(f"Authentication failed for user {username}")
    return user


def create_test_users(customer_id: str, users: List[Dict]) -> None:
    """
    Create test users for development and testing.

    Parameters
    ----------
    customer_id : str
        Customer identifier
    users : List[Dict]
        List of user dictionaries with username, password, etc.
    """
    # Get the customer's root directory
    customer_root = customer_path(customer_id)
    config_dir = customer_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    users_file = config_dir / "users.json"
    credentials_file = config_dir / "credentials.json"

    # Create user records
    users_json = []
    credentials = {}

    for i, user_dict in enumerate(users):
        username = user_dict["username"]
        password = user_dict["password"]

        # Create user record
        user_id = f"{customer_id}_{username}_{int(time.time())}"
        user_record = {
            "id": user_id,
            "username": username,
            "email": user_dict.get("email", f"{username}@example.com"),
            "customer_id": customer_id,
            "full_name": user_dict.get("full_name", username.title()),
            "roles": user_dict.get("roles", ["user"])
        }
        users_json.append(user_record)

        # Create credential record
        credentials[username] = _user_store._hash_password(password)

    # Save to files
    with open(users_file, "w") as f:
        json.dump(users_json, f, indent=2)

    with open(credentials_file, "w") as f:
        json.dump(credentials, f, indent=2)

    logger.info(f"Created {len(users)} test users for customer {customer_id}")
