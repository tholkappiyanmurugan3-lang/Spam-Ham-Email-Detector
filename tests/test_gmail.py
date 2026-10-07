"""
tests/test_gmail.py
===================
Unit tests for Gmail header decoding and body extraction logic.
"""

import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pytest

from spam_detector.gmail import _decode_header_value, _extract_body_from_email_message


def test_decode_header_plain():
    assert _decode_header_value("Simple Subject") == "Simple Subject"
    assert _decode_header_value(None) == ""


def test_decode_header_encoded():
    # Encoded RFC 2047 subject
    encoded = "=?utf-8?B?V2luIGEgbWlsbGlvbiBkb2xsYXJzIQ==?="
    assert _decode_header_value(encoded) == "Win a million dollars!"


def test_extract_body_single_part():
    msg = MIMEText("This is a single part plain text body.", "plain", "utf-8")
    body = _extract_body_from_email_message(msg)
    assert "This is a single part plain text body." in body


def test_extract_body_multipart():
    msg = MIMEMultipart("alternative")
    part_text = MIMEText("Plain text version.", "plain", "utf-8")
    part_html = MIMEText("<p>HTML version.</p>", "html", "utf-8")
    msg.attach(part_text)
    msg.attach(part_html)

    body = _extract_body_from_email_message(msg)
    assert "Plain text version." in body
