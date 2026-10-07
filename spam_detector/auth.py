"""
spam_detector.auth
==================
Enterprise-style user authentication, password hashing, and session management.

Security design:
* Passwords are salted and hashed using PBKDF2-HMAC-SHA256 (100,000 iterations).
* Verification uses secrets.compare_digest to prevent timing attacks.
* SQLite user table stores credentials, roles, timestamps, and connected Gmail settings.
* Generates a default admin account on first startup for immediate access.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger
from spam_detector.config import cfg
from spam_detector.db import _get_connection


@dataclass
class User:
    id: int
    username: str
    email: str
    full_name: str
    role: str
    created_at: str
    last_login: Optional[str] = None
    gmail_address: Optional[str] = None
    gmail_app_password: Optional[str] = None

    @property
    def has_gmail_connected(self) -> bool:
        return bool(self.gmail_address and self.gmail_app_password)


def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Generate a cryptographic hash and salt using PBKDF2-HMAC-SHA256."""
    if not salt:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()
    return pw_hash, salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify password against stored hash using constant-time comparison."""
    test_hash, _ = _hash_password(password, salt)
    return secrets.compare_digest(test_hash, stored_hash)


def init_auth_db(db_path: Optional[Path] = None) -> None:
    """Create the users table if it does not already exist, and create default admin."""
    conn = _get_connection(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                username           TEXT UNIQUE NOT NULL,
                email              TEXT UNIQUE NOT NULL,
                password_hash      TEXT NOT NULL,
                salt               TEXT NOT NULL,
                full_name          TEXT,
                role               TEXT NOT NULL DEFAULT 'analyst',
                created_at         TEXT NOT NULL,
                last_login         TEXT,
                gmail_address      TEXT,
                gmail_app_password TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

        # Ensure existing databases get the new columns if migrated
        try:
            conn.execute("ALTER TABLE users ADD COLUMN gmail_address TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists

        try:
            conn.execute("ALTER TABLE users ADD COLUMN gmail_app_password TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists

        # Create default demo admin if users table is empty
        cur = conn.execute("SELECT COUNT(*) AS count FROM users")
        if cur.fetchone()["count"] == 0:
            pw_hash, salt = _hash_password("admin123")
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, salt, full_name, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("admin", "admin@spamshield.ai", pw_hash, salt, "System Administrator", "admin", now),
            )
            logger.info("Created default administrator user: username='admin'")

    conn.close()


def update_user_gmail_credentials(
    user_id: int,
    gmail_address: str,
    gmail_app_password: str,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Link or update Gmail credentials for a registered user.
    """
    clean_addr = gmail_address.strip()
    clean_pass = gmail_app_password.replace(" ", "").strip()

    conn = _get_connection(db_path)
    with conn:
        conn.execute(
            """
            UPDATE users
            SET gmail_address = ?, gmail_app_password = ?
            WHERE id = ?
            """,
            (clean_addr, clean_pass, user_id),
        )
    conn.close()
    logger.info(f"Updated Gmail credentials for user id={user_id}")
    return True


def register_user(
    username: str,
    email: str,
    password: str,
    full_name: str = "",
    role: str = "analyst",
    gmail_address: Optional[str] = None,
    gmail_app_password: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> tuple[bool, str]:
    """
    Register a new user account with optional Gmail linkage.

    Returns
    -------
    tuple[bool, str]
        (Success, message)
    """
    clean_username = username.strip().lower()
    clean_email = email.strip().lower()

    if not clean_username or len(clean_username) < 3:
        return False, "Username must be at least 3 characters long."
    if "@" not in clean_email or "." not in clean_email:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    clean_gmail = gmail_address.strip() if gmail_address else None
    clean_gmail_pass = gmail_app_password.replace(" ", "").strip() if gmail_app_password else None

    pw_hash, salt = _hash_password(password)
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_connection(db_path)

    try:
        with conn:
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, salt, full_name, role, created_at, gmail_address, gmail_app_password)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (clean_username, clean_email, pw_hash, salt, full_name.strip() or clean_username.title(), role, now, clean_gmail, clean_gmail_pass),
            )
        conn.close()
        logger.info(f"Successfully registered user: {clean_username}")
        return True, "Account created successfully! You can now log in."
    except sqlite3.IntegrityError as err:
        conn.close()
        err_msg = str(err).lower()
        if "username" in err_msg:
            return False, f"Username '{clean_username}' is already taken."
        if "email" in err_msg:
            return False, f"Email '{clean_email}' is already registered."
        return False, "An account with these details already exists."
    except Exception as exc:
        conn.close()
        return False, f"Registration failed: {exc}"


def authenticate_user(
    username_or_email: str,
    password: str,
    db_path: Optional[Path] = None,
) -> tuple[Optional[User], str]:
    """
    Validate user credentials and return User object with connected Gmail info.

    Returns
    -------
    tuple[Optional[User], str]
        (User object or None, status message)
    """
    ident = username_or_email.strip().lower()
    conn = _get_connection(db_path)
    cur = conn.execute(
        "SELECT * FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?",
        (ident, ident),
    )
    row = cur.fetchone()

    if not row:
        conn.close()
        return None, "Invalid username or password."

    if not _verify_password(password, row["password_hash"], row["salt"]):
        conn.close()
        return None, "Invalid username or password."

    # Update last login timestamp
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, row["id"]))
    conn.close()

    # Safely extract optional columns
    keys = row.keys()
    gmail_addr = row["gmail_address"] if "gmail_address" in keys else None
    gmail_pass = row["gmail_app_password"] if "gmail_app_password" in keys else None

    user = User(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        full_name=row["full_name"] or row["username"].title(),
        role=row["role"],
        created_at=row["created_at"],
        last_login=now,
        gmail_address=gmail_addr,
        gmail_app_password=gmail_pass,
    )
    logger.info(f"User authenticated: {user.username} (role: {user.role})")
    return user, "Authentication successful."


# ---------------------------------------------------------------------------
# Password Reset (Forgot Password)
# ---------------------------------------------------------------------------

def _init_reset_tokens_table(db_path: Optional[Path] = None) -> None:
    """Ensure the password_reset_tokens table exists."""
    conn = _get_connection(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                email      TEXT NOT NULL,
                token      TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used       INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reset_token ON password_reset_tokens(token)"
        )
    conn.close()


def create_reset_token(
    email: str,
    db_path: Optional[Path] = None,
    expiry_minutes: int = 15,
) -> tuple[bool, str, str]:
    """
    Generate a 6-digit OTP reset token for the given email.

    Returns
    -------
    tuple[bool, str, str]
        (success, message, token)
        token is empty string on failure.
    """
    _init_reset_tokens_table(db_path)
    clean_email = email.strip().lower()

    conn = _get_connection(db_path)
    cur = conn.execute(
        "SELECT id FROM users WHERE LOWER(email) = ?", (clean_email,)
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        # Return generic message to prevent email enumeration
        return False, "If that email is registered, a reset code will be sent.", ""

    # Invalidate any previous unused tokens for this email
    conn = _get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE password_reset_tokens SET used = 1 WHERE email = ? AND used = 0",
            (clean_email,),
        )
    conn.close()

    # Generate a 6-digit numeric OTP (easy to type)
    token = str(secrets.randbelow(900000) + 100000)  # 100000–999999

    from datetime import timedelta
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=expiry_minutes)

    conn = _get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO password_reset_tokens (email, token, created_at, expires_at, used)
            VALUES (?, ?, ?, ?, 0)
            """,
            (clean_email, token, now.isoformat(), expires.isoformat()),
        )
    conn.close()

    logger.info(f"Reset token created for {clean_email} (expires {expiry_minutes} min)")
    return True, "Reset code generated successfully.", token


def validate_and_reset_password(
    email: str,
    token: str,
    new_password: str,
    db_path: Optional[Path] = None,
) -> tuple[bool, str]:
    """
    Validate the OTP token and update the user's password.

    Returns
    -------
    tuple[bool, str]
        (success, message)
    """
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters long."

    clean_email = email.strip().lower()
    clean_token = token.strip()

    _init_reset_tokens_table(db_path)
    conn = _get_connection(db_path)
    cur = conn.execute(
        """
        SELECT * FROM password_reset_tokens
        WHERE email = ? AND token = ? AND used = 0
        ORDER BY created_at DESC LIMIT 1
        """,
        (clean_email, clean_token),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return False, "Invalid or expired reset code. Please request a new one."

    # Check expiry
    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        return False, "This reset code has expired (15 minutes). Please request a new one."

    # Update password
    new_hash, new_salt = _hash_password(new_password)
    conn = _get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE LOWER(email) = ?",
            (new_hash, new_salt, clean_email),
        )
        # Mark token as used
        conn.execute(
            "UPDATE password_reset_tokens SET used = 1 WHERE id = ?",
            (row["id"],),
        )
    conn.close()

    logger.info(f"Password successfully reset for {clean_email}")
    return True, "Password reset successfully! You can now log in with your new password."


def send_reset_email_smtp(
    to_email: str,
    token: str,
    sender_gmail: str,
    sender_app_password: str,
) -> tuple[bool, str]:
    """
    Send the password reset OTP via Gmail SMTP (STARTTLS on port 587).

    Parameters
    ----------
    to_email : str
        Recipient email address (the user who forgot their password).
    token : str
        6-digit OTP code to include in the email.
    sender_gmail : str
        Gmail address to send from (must have App Password enabled).
    sender_app_password : str
        16-character Google App Password for the sender account.

    Returns
    -------
    tuple[bool, str]
        (success, message)
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    subject = "SpamShield — Password Reset Code"

    html_body = f"""
    <div style="font-family:Arial,sans-serif; max-width:480px; margin:auto;
                border:1px solid #e5e7eb; border-radius:12px; overflow:hidden;">
      <div style="background:#1e293b; padding:24px; text-align:center;">
        <h1 style="color:#38bdf8; margin:0; font-size:1.5rem;">🛡️ SpamShield</h1>
        <p style="color:#94a3b8; margin:4px 0 0;">Email Security Platform</p>
      </div>
      <div style="padding:32px;">
        <h2 style="color:#1e293b; margin-top:0;">Password Reset Request</h2>
        <p style="color:#374151;">We received a request to reset your SpamShield password.
           Use the code below to set a new password.</p>
        <div style="background:#f0f9ff; border:2px solid #38bdf8; border-radius:8px;
                    padding:20px; text-align:center; margin:24px 0;">
          <p style="margin:0; color:#64748b; font-size:0.85rem; letter-spacing:1px;">
            YOUR RESET CODE
          </p>
          <p style="margin:8px 0 0; font-size:2.5rem; font-weight:900;
                    letter-spacing:12px; color:#0f172a;">{token}</p>
        </div>
        <p style="color:#6b7280; font-size:0.85rem;">
          ⏱️ This code expires in <strong>15 minutes</strong>.<br>
          🔒 If you didn't request a password reset, you can ignore this email.
        </p>
      </div>
      <div style="background:#f8fafc; padding:16px; text-align:center;
                  border-top:1px solid #e5e7eb;">
        <p style="color:#9ca3af; font-size:0.75rem; margin:0;">
          SpamShield AI Email Security — Do not reply to this email.
        </p>
      </div>
    </div>
    """

    plain_body = f"""SpamShield — Password Reset

Your reset code is: {token}

This code expires in 15 minutes.
If you didn't request this, ignore this email.
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"SpamShield Security <{sender_gmail}>"
    msg["To"] = to_email
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(sender_gmail, sender_app_password.replace(" ", ""))
            server.sendmail(sender_gmail, to_email, msg.as_string())
        logger.info(f"Reset email sent to {to_email}")
        return True, f"Reset code sent to {to_email}"
    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed. Check the sender Gmail App Password."
    except Exception as exc:
        logger.error(f"Failed to send reset email: {exc}")
        return False, f"Failed to send email: {exc}"
