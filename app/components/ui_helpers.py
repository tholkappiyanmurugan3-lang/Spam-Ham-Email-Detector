"""
app/components/ui_helpers.py
============================
Shared Streamlit UI helper functions.
"""

import streamlit as st


def confidence_badge(confidence: float, threshold: float = 0.85) -> str:
    """Return an HTML badge string for a confidence score."""
    if confidence >= threshold:
        color, label = "#22C55E", "High Confidence"
    elif confidence >= 0.70:
        color, label = "#F59E0B", "Medium Confidence"
    else:
        color, label = "#EF4444", "Low Confidence"
    return (
        f'<span style="background:{color}; color:white; padding:3px 10px; '
        f'border-radius:12px; font-size:0.8em;">{label} ({confidence:.0%})</span>'
    )


def spam_badge(label: str) -> str:
    """Return a styled HTML badge for SPAM or HAM."""
    if label == "SPAM":
        return '<span style="background:#EF4444;color:white;padding:4px 12px;border-radius:12px;">🚨 SPAM</span>'
    return '<span style="background:#22C55E;color:white;padding:4px 12px;border-radius:12px;">✅ HAM</span>'
