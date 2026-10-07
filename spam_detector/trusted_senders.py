"""
spam_detector.trusted_senders
==============================
Whitelist of domains whose emails should never be classified as SPAM.

These are high-volume legitimate transactional senders that the SMS-trained
model may incorrectly flag due to URL-heavy or formal language patterns.

The whitelist is intentionally conservative — only verified, globally
recognised domains are included. User-defined additions are supported via
the TRUSTED_DOMAINS environment variable (comma-separated).
"""

from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# Core verified trusted domains
# ---------------------------------------------------------------------------
_CORE_TRUSTED_DOMAINS: set[str] = {
    # Google services
    "google.com",
    "accounts.google.com",
    "gmail.com",
    "googlemail.com",
    "youtube.com",
    "google.co.in",

    # Microsoft / Office 365
    "microsoft.com",
    "outlook.com",
    "office.com",
    "live.com",
    "hotmail.com",
    "microsoftonline.com",
    "azure.com",

    # Apple
    "apple.com",
    "icloud.com",
    "appleid.apple.com",

    # Amazon / AWS
    "amazon.com",
    "amazon.in",
    "aws.amazon.com",
    "ses.amazonaws.com",

    # Social / professional
    "linkedin.com",
    "github.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",

    # Indian government / services
    "gov.in",
    "nic.in",
    "npci.org.in",

    # Banking (commonly impersonated — include only verified ones)
    "hdfcbank.com",
    "icicibank.com",
    "sbi.co.in",
    "axisbank.com",
    "kotak.com",

    # Payment
    "razorpay.com",
    "paytm.com",
    "phonepe.com",
    "stripe.com",
    "paypal.com",
}

_EMAIL_DOMAIN_RE = re.compile(r"@([\w.\-]+)", re.IGNORECASE)


def _load_trusted_domains() -> set[str]:
    """Load core domains plus any user-defined ones from the environment."""
    domains = set(_CORE_TRUSTED_DOMAINS)
    extra = os.environ.get("TRUSTED_DOMAINS", "").strip()
    if extra:
        for d in extra.split(","):
            d = d.strip().lower()
            if d:
                domains.add(d)
    return domains


# Module-level set (loaded once)
TRUSTED_DOMAINS: set[str] = _load_trusted_domains()


def is_trusted_sender(sender: str) -> bool:
    """
    Return True if the sender email address belongs to a trusted domain.

    Parameters
    ----------
    sender : str
        Raw "From" header value, e.g. ``"Google <noreply@google.com>"``
        or plain ``"noreply@google.com"``.

    Returns
    -------
    bool
        True if the domain (or any parent domain) is in the trusted list.

    Examples
    --------
    >>> is_trusted_sender("Google <google-noreply@google.com>")
    True
    >>> is_trusted_sender("prize@win-free-iphone.xyz")
    False
    """
    if not sender:
        return False

    matches = _EMAIL_DOMAIN_RE.findall(sender.lower())
    if not matches:
        return False

    domain = matches[-1]  # Take the last match to avoid display-name spoofing

    # Exact match or subdomain match
    if domain in TRUSTED_DOMAINS:
        return True

    # Check parent domain (e.g. "accounts.google.com" → "google.com")
    parts = domain.split(".")
    for i in range(1, len(parts) - 1):
        parent = ".".join(parts[i:])
        if parent in TRUSTED_DOMAINS:
            return True

    return False
