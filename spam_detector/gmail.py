"""
spam_detector.gmail
===================
Fetch and classify live emails directly from Gmail via IMAP.

Authentication
--------------
Uses Gmail App Passwords (recommended and secure).
Does NOT require your main Google account password or complex OAuth setup.

Setup steps:
1. Enable 2-Step Verification in Google Account:
   https://myaccount.google.com/security
2. Generate an App Password:
   https://myaccount.google.com/apppasswords
   - Select App: "Mail"
   - Select Device: "Windows Computer" (or "Other")
   - Copy the 16-character generated password (e.g. "abcd efgh ijkl mnop")
3. Set GMAIL_USER and GMAIL_APP_PASSWORD in .env or pass directly.
"""

from __future__ import annotations

import email
from email.header import decode_header
import imaplib
import os
from typing import Optional, Any
from dataclasses import dataclass

from loguru import logger

from spam_detector.predict import Predictor, PredictionResult
from spam_detector.db import log_prediction


IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993


@dataclass
class GmailMessage:
    msg_id: str
    sender: str
    subject: str
    date: str
    body: str


def _decode_header_value(header_val: Optional[str]) -> str:
    """Decode RFC 2047 encoded email headers."""
    if not header_val:
        return ""
    decoded_fragments = decode_header(header_val)
    parts = []
    for fragment, encoding in decoded_fragments:
        if isinstance(fragment, bytes):
            enc = encoding or "utf-8"
            try:
                parts.append(fragment.decode(enc, errors="replace"))
            except Exception:
                parts.append(fragment.decode("latin-1", errors="replace"))
        else:
            parts.append(str(fragment))
    return "".join(parts)


def _extract_body_from_email_message(msg: email.message.Message) -> str:
    """Extract plain text or HTML body from email.message.Message."""
    body_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            # Ignore attachments
            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        return payload.decode(charset, errors="replace")
                    except Exception:
                        return payload.decode("latin-1", errors="replace")
            elif content_type == "text/html" and not body_text:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        body_text = payload.decode(charset, errors="replace")
                    except Exception:
                        body_text = payload.decode("latin-1", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                body_text = payload.decode(charset, errors="replace")
            except Exception:
                body_text = payload.decode("latin-1", errors="replace")

    return body_text


def fetch_gmail_emails(
    username: str,
    app_password: str,
    folder: str = "INBOX",
    limit: int = 10,
    unread_only: bool = False,
) -> list[GmailMessage]:
    """
    Connect to Gmail via IMAP and retrieve recent messages.

    Parameters
    ----------
    username : str
        Gmail email address (e.g. user@gmail.com).
    app_password : str
        16-character Google App Password (spaces will be stripped).
    folder : str
        Mailbox folder to inspect (default: "INBOX").
    limit : int
        Maximum number of emails to retrieve.
    unread_only : bool
        If True, only fetches unread messages.

    Returns
    -------
    list[GmailMessage]
        Parsed email messages sorted newest first.
    """
    clean_password = app_password.replace(" ", "")
    logger.info(f"Connecting to {IMAP_SERVER}:{IMAP_PORT} for user {username}")

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(username, clean_password)
    except imaplib.IMAP4.error as exc:
        logger.error(f"Gmail IMAP authentication failed: {exc}")
        raise ValueError(
            f"Gmail authentication failed: {exc}. "
            "Please ensure you are using a 16-character Google App Password "
            "(not your standard Google account password)."
        ) from exc

    status, _ = mail.select(f'"{folder}"', readonly=True)
    if status != "OK":
        mail.logout()
        raise ValueError(f"Could not open Gmail folder '{folder}'.")

    criterion = "UNSEEN" if unread_only else "ALL"
    status, data = mail.search(None, criterion)
    if status != "OK" or not data or not data[0]:
        logger.info(f"No emails found matching criterion '{criterion}'.")
        mail.logout()
        return []

    msg_nums = data[0].split()
    # Newest emails are at the end of the list
    selected_nums = msg_nums[-limit:]
    selected_nums.reverse()

    messages: list[GmailMessage] = []
    for num in selected_nums:
        status, response_data = mail.fetch(num, "(RFC822)")
        if status != "OK" or not response_data:
            continue

        raw_email = response_data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject = _decode_header_value(msg.get("Subject"))
        sender = _decode_header_value(msg.get("From"))
        date_str = _decode_header_value(msg.get("Date"))
        body = _extract_body_from_email_message(msg)

        messages.append(
            GmailMessage(
                msg_id=num.decode("utf-8", errors="ignore"),
                sender=sender,
                subject=subject,
                date=date_str,
                body=body,
            )
        )

    mail.logout()
    logger.info(f"Successfully retrieved {len(messages)} messages from Gmail.")
    return messages


def scan_gmail(
    predictor: Predictor,
    username: str,
    app_password: str,
    folder: str = "INBOX",
    limit: int = 10,
    unread_only: bool = False,
    log_to_db: bool = True,
) -> list[dict[str, Any]]:
    """
    Fetch emails from Gmail, classify them using Predictor, and optionally log.

    Returns
    -------
    list[dict[str, Any]]
        List of dicts containing email details + prediction results.
    """
    emails = fetch_gmail_emails(
        username=username,
        app_password=app_password,
        folder=folder,
        limit=limit,
        unread_only=unread_only,
    )

    results = []
    for mail in emails:
        # Include Subject in the evaluated text as subject lines are strong spam signals
        eval_text = f"Subject: {mail.subject}\n\n{mail.body}"
        pred = predictor.predict(eval_text)

        if log_to_db:
            log_prediction(pred, source="gmail")

        results.append({
            "id": mail.msg_id,
            "sender": mail.sender,
            "subject": mail.subject,
            "date": mail.date,
            "raw_body": mail.body,
            "eval_text": eval_text,
            "label": pred.label,
            "confidence": pred.confidence,
            "spam_prob": pred.spam_prob,
            "ham_prob": pred.ham_prob,
            "is_spam": pred.is_spam,
            "is_uncertain": pred.is_uncertain,
        })

    return results



if __name__ == "__main__":
    import argparse
    from spam_detector.config import cfg

    parser = argparse.ArgumentParser(description="Scan Gmail inbox for spam.")
    parser.add_argument("--email", type=str, default=os.getenv("GMAIL_USER"), help="Gmail address")
    parser.add_argument("--password", type=str, default=os.getenv("GMAIL_APP_PASSWORD"), help="Gmail 16-char App Password")
    parser.add_argument("--limit", type=int, default=5, help="Number of emails to fetch (default: 5)")
    parser.add_argument("--unread", action="store_true", help="Fetch unread emails only")

    args = parser.parse_args()

    if not args.email or not args.password:
        print("Error: Please provide --email and --password (or set GMAIL_USER & GMAIL_APP_PASSWORD in .env)")
        exit(1)

    print(f"Loading model from {cfg.model_dir}...")
    predictor = Predictor()
    print(f"Scanning latest {args.limit} emails for {args.email}...")
    scanned = scan_gmail(predictor, args.email, args.password, limit=args.limit, unread_only=args.unread)

    print("\n--- Scan Results ---")
    for item in scanned:
        badge = "[SPAM]" if item["is_spam"] else "[HAM]"
        print(f"{badge} ({item['confidence']:.1%}) From: {item['sender']} | Subject: {item['subject']}")
