"""
spam_detector.gmail_diagnostics
================================
Step-by-step connection diagnostic tool for Gmail IMAP.
"""

from __future__ import annotations
import imaplib
import socket
from typing import Optional

IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993


def diagnose_gmail_connection(username: str, app_password: str) -> list[dict]:
    """
    Run step-by-step IMAP diagnostics and return a list of result steps.

    Returns
    -------
    list[dict]  -- each entry: {"step": str, "ok": bool, "message": str}
    """
    steps = []
    clean_password = app_password.replace(" ", "")

    # ---------------------------------------------------------------
    # Step 1: basic input validation
    # ---------------------------------------------------------------
    if "@" not in username or "." not in username:
        steps.append({
            "step": "Input Validation",
            "ok": False,
            "message": f"'{username}' does not look like a valid email address.",
        })
        return steps

    if len(clean_password) != 16:
        steps.append({
            "step": "Input Validation",
            "ok": False,
            "message": (
                f"Password is {len(clean_password)} characters after removing spaces. "
                "A Google App Password must be exactly 16 characters. "
                "It should look like 'abcd efgh ijkl mnop' (4 groups of 4 letters)."
            ),
        })
        return steps

    steps.append({
        "step": "Input Validation",
        "ok": True,
        "message": f"Email '{username}' is valid and password is exactly 16 characters.",
    })

    # ---------------------------------------------------------------
    # Step 2: DNS / network reachability
    # ---------------------------------------------------------------
    try:
        ip = socket.gethostbyname(IMAP_SERVER)
        steps.append({
            "step": "DNS Resolution",
            "ok": True,
            "message": f"Resolved {IMAP_SERVER} → {ip}",
        })
    except socket.gaierror as exc:
        steps.append({
            "step": "DNS Resolution",
            "ok": False,
            "message": f"Cannot resolve {IMAP_SERVER}: {exc}. Check your internet connection.",
        })
        return steps

    # ---------------------------------------------------------------
    # Step 3: TCP connection
    # ---------------------------------------------------------------
    try:
        sock = socket.create_connection((IMAP_SERVER, IMAP_PORT), timeout=8)
        sock.close()
        steps.append({
            "step": "TCP Connection",
            "ok": True,
            "message": f"Successfully opened TCP connection to {IMAP_SERVER}:{IMAP_PORT}.",
        })
    except (socket.timeout, OSError) as exc:
        steps.append({
            "step": "TCP Connection",
            "ok": False,
            "message": f"Cannot reach {IMAP_SERVER}:{IMAP_PORT}: {exc}. Firewall or port 993 may be blocked.",
        })
        return steps

    # ---------------------------------------------------------------
    # Step 4: SSL handshake
    # ---------------------------------------------------------------
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        steps.append({
            "step": "SSL Handshake",
            "ok": True,
            "message": "SSL/TLS connection established with Gmail's IMAP server.",
        })
    except Exception as exc:
        steps.append({
            "step": "SSL Handshake",
            "ok": False,
            "message": f"SSL connection failed: {exc}",
        })
        return steps

    # ---------------------------------------------------------------
    # Step 5: IMAP login
    # ---------------------------------------------------------------
    try:
        mail.login(username, clean_password)
        steps.append({
            "step": "IMAP Authentication",
            "ok": True,
            "message": "Login successful! Credentials are valid.",
        })
    except imaplib.IMAP4.error as exc:
        err_str = str(exc).lower()

        if "authenticationfailed" in err_str or "invalid credentials" in err_str:
            tip = (
                "Authentication rejected by Google. This means:\n"
                "• The password entered is NOT a 16-character App Password — it may be your regular Google password.\n"
                "• OR: Your account is a **Google Workspace/school/company account** — "
                "the admin must enable App Passwords in the Google Workspace Admin Console.\n"
                "• OR: The App Password was recently revoked — generate a new one at "
                "https://myaccount.google.com/apppasswords"
            )
        elif "too many" in err_str:
            tip = "Too many failed login attempts. Wait 30 minutes before trying again."
        else:
            tip = f"Raw error: {exc}"

        steps.append({
            "step": "IMAP Authentication",
            "ok": False,
            "message": tip,
        })
        return steps

    # ---------------------------------------------------------------
    # Step 6: mailbox access
    # ---------------------------------------------------------------
    try:
        status, data = mail.select('"INBOX"', readonly=True)
        if status == "OK":
            msg_count = data[0].decode() if data[0] else "?"
            steps.append({
                "step": "Mailbox Access",
                "ok": True,
                "message": f"INBOX opened successfully. Contains ~{msg_count} message(s).",
            })
        else:
            steps.append({
                "step": "Mailbox Access",
                "ok": False,
                "message": f"Could not open INBOX. Status: {status}",
            })
    except Exception as exc:
        steps.append({
            "step": "Mailbox Access",
            "ok": False,
            "message": f"Error accessing INBOX: {exc}",
        })
    finally:
        try:
            mail.logout()
        except Exception:
            pass

    return steps
