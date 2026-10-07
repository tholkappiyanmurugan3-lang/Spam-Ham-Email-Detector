"""
tests/test_auth.py
==================
Unit tests for user registration, authentication, and password hashing.
"""

import tempfile
from pathlib import Path
import pytest

from spam_detector.auth import (
    _hash_password,
    _verify_password,
    init_auth_db,
    register_user,
    authenticate_user,
    update_user_gmail_credentials,
)


@pytest.fixture
def auth_db(tmp_path):
    """Create a temporary SQLite database with users table."""
    db_file = tmp_path / "test_auth.db"
    init_auth_db(db_path=db_file)
    return db_file


def test_hash_and_verify():
    pw = "SecretPassword123"
    pw_hash, salt = _hash_password(pw)
    assert _verify_password(pw, pw_hash, salt) is True
    assert _verify_password("WrongPassword", pw_hash, salt) is False


def test_default_admin_created(auth_db):
    user, msg = authenticate_user("admin", "admin123", db_path=auth_db)
    assert user is not None
    assert user.username == "admin"
    assert user.role == "admin"


def test_register_and_authenticate(auth_db):
    success, msg = register_user(
        username="alice",
        email="alice@example.com",
        password="mypassword",
        full_name="Alice Smith",
        role="analyst",
        db_path=auth_db,
    )
    assert success is True

    # Authenticate via username
    user_by_name, _ = authenticate_user("alice", "mypassword", db_path=auth_db)
    assert user_by_name is not None
    assert user_by_name.email == "alice@example.com"
    assert user_by_name.full_name == "Alice Smith"

    # Authenticate via email
    user_by_email, _ = authenticate_user("alice@example.com", "mypassword", db_path=auth_db)
    assert user_by_email is not None
    assert user_by_email.username == "alice"


def test_register_with_gmail(auth_db):
    success, _ = register_user(
        username="guser",
        email="guser@example.com",
        password="password123",
        full_name="Gmail User",
        gmail_address="myemail@gmail.com",
        gmail_app_password="abcd efgh ijkl mnop",
        db_path=auth_db,
    )
    assert success is True

    user, _ = authenticate_user("guser", "password123", db_path=auth_db)
    assert user is not None
    assert user.gmail_address == "myemail@gmail.com"
    assert user.gmail_app_password == "abcdefghijklmnop"
    assert user.has_gmail_connected is True


def test_update_user_gmail(auth_db):
    register_user("bob", "bob@example.com", "password123", db_path=auth_db)
    user, _ = authenticate_user("bob", "password123", db_path=auth_db)
    assert user.has_gmail_connected is False

    update_user_gmail_credentials(user.id, "bob_gmail@gmail.com", "wxyz abcd 1234 5678", db_path=auth_db)
    updated_user, _ = authenticate_user("bob", "password123", db_path=auth_db)
    assert updated_user.gmail_address == "bob_gmail@gmail.com"
    assert updated_user.gmail_app_password == "wxyzabcd12345678"
    assert updated_user.has_gmail_connected is True


def test_register_duplicate_username(auth_db):
    register_user("bob2", "bob2@example.com", "password123", db_path=auth_db)
    success, msg = register_user("bob2", "bob_diff@example.com", "password123", db_path=auth_db)
    assert success is False
    assert "taken" in msg.lower()


def test_register_duplicate_email(auth_db):
    register_user("charlie", "charlie@example.com", "password123", db_path=auth_db)
    success, msg = register_user("charlie2", "charlie@example.com", "password123", db_path=auth_db)
    assert success is False
    assert "already registered" in msg.lower()


def test_authenticate_invalid_credentials(auth_db):
    user, msg = authenticate_user("nonexistent", "password", db_path=auth_db)
    assert user is None
    assert "invalid" in msg.lower()

    user, msg = authenticate_user("admin", "wrongpassword", db_path=auth_db)
    assert user is None
    assert "invalid" in msg.lower()
