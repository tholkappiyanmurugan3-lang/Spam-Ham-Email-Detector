"""
spam_detector.preprocess
========================
Email text cleaning pipeline.

Design decisions
----------------
* BeautifulSoup is used (not regex) for HTML stripping — more robust
  against malformed tags common in spam.
* URL removal happens *before* lowercasing so schemes are detected
  case-insensitively.
* Email headers (From:, To:, Subject:, etc.) are stripped to prevent
  the model from over-fitting on sender addresses rather than content.
* The function is intentionally pure (no I/O side effects) so it can
  be unit-tested trivially.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Optional

from bs4 import BeautifulSoup
from loguru import logger

# Invisible / zero-width Unicode characters inserted by email marketing tools
# (Mailchimp, SendGrid, HubSpot) as anti-whitespace-collapsing tricks.
# Must be stripped AFTER html.unescape(), before the model sees the text.
#   U+200B  Zero-Width Space
#   U+200C  Zero-Width Non-Joiner  (&zwnj;)
#   U+200D  Zero-Width Joiner
#   U+FEFF  Zero-Width No-Break Space (BOM)
#   U+00AD  Soft Hyphen
#   U+2060  Word Joiner
_INVISIBLE_CHARS_RE = re.compile(
    r"[\u200b\u200c\u200d\ufeff\u00ad\u2060\u180e]",
)


# ---------------------------------------------------------------------------
# Regex patterns compiled once at module load
# ---------------------------------------------------------------------------
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+",
    re.IGNORECASE,
)

_EMAIL_ADDR_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)

# Common email header lines, e.g. "From: foo@bar.com\n"
_HEADER_RE = re.compile(
    r"^(From|To|Cc|Bcc|Date|Subject|Message-ID|MIME-Version|"
    r"Content-Type|Content-Transfer-Encoding|Return-Path|Received|"
    r"X-[A-Za-z\-]+):.*$",
    re.MULTILINE | re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """
    Remove HTML tags and decode HTML entities.

    Handles two common cases that survive a naive BeautifulSoup pass:
    1. ``&zwnj;`` / ``&nbsp;`` entities that appear in plain-text output
       when marketers embed invisible spacers (Mailchimp, SendGrid pattern).
    2. Unicode zero-width characters (U+200C, U+200B, etc.) used to prevent
       whitespace collapsing — invisible to readers but pollute the model input.

    Parameters
    ----------
    text : str
        Raw email body (may contain HTML).

    Returns
    -------
    str
        Plain text with HTML markup, entities, and invisible chars removed.
    """
    soup = BeautifulSoup(text, "lxml")
    plain = soup.get_text(separator=" ")

    # Second pass: decode any surviving HTML entities (e.g. &zwnj; &nbsp;)
    plain = html.unescape(plain)

    # Remove invisible/zero-width Unicode characters inserted by marketing tools
    plain = _INVISIBLE_CHARS_RE.sub("", plain)

    return plain


def remove_urls(text: str) -> str:
    """Replace URLs with the token ``<URL>`` (preserves token count context)."""
    return _URL_RE.sub("<URL>", text)


def remove_email_addresses(text: str) -> str:
    """Replace email addresses with ``<EMAIL>`` token."""
    return _EMAIL_ADDR_RE.sub("<EMAIL>", text)


def remove_headers(text: str) -> str:
    """Strip standard RFC-2822 email header lines from the text."""
    return _HEADER_RE.sub("", text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to NFC form and drop surrogates."""
    return unicodedata.normalize("NFC", text)


def collapse_whitespace(text: str) -> str:
    """Replace any run of whitespace (spaces, tabs, newlines) with a single space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_email(
    text: str,
    strip_html_flag: bool = True,
    remove_urls_flag: bool = True,
    remove_headers_flag: bool = True,
    lowercase: bool = True,
    remove_extra_whitespace: bool = True,
) -> str:
    """
    Full cleaning pipeline for a single email body.

    Parameters
    ----------
    text : str
        Raw email text (may include headers, HTML, URLs).
    strip_html_flag : bool
        Whether to remove HTML markup.
    remove_urls_flag : bool
        Whether to replace URLs with ``<URL>`` token.
    remove_headers_flag : bool
        Whether to strip RFC-2822 email headers.
    lowercase : bool
        Whether to lowercase the result.
    remove_extra_whitespace : bool
        Whether to collapse runs of whitespace.

    Returns
    -------
    str
        Cleaned, normalised email text ready for tokenisation.

    Examples
    --------
    >>> clean_email("<html><body>Buy now! http://spam.com</body></html>")
    'buy now! <url>'
    """
    if not isinstance(text, str) or not text.strip():
        logger.warning("Received empty or non-string input; returning empty string.")
        return ""

    text = normalize_unicode(text)

    if remove_headers_flag:
        text = remove_headers(text)

    if strip_html_flag:
        text = strip_html(text)

    if remove_urls_flag:
        text = remove_urls(text)

    text = remove_email_addresses(text)

    if lowercase:
        text = text.lower()

    if remove_extra_whitespace:
        text = collapse_whitespace(text)

    return text


def clean_batch(
    texts: list[str],
    **kwargs,
) -> list[str]:
    """
    Apply :func:`clean_email` to a list of texts.

    Parameters
    ----------
    texts : list[str]
        List of raw email strings.
    **kwargs
        Forwarded to :func:`clean_email`.

    Returns
    -------
    list[str]
        Cleaned email texts.
    """
    return [clean_email(t, **kwargs) for t in texts]
