"""
app/components/ui_theme.py
==========================
Clean, attractive, minimalist UI theme for SpamShield.
Designed to be lightweight, modern, and compatible with both light and dark modes.
"""

import streamlit as st

THEME_CSS = """
<style>
/* Subtle typography & smooth layout */
body, [class*="css"] {
    letter-spacing: -0.01em;
}

/* Polished clean buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-1px);
}

/* Clean container card padding */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
}

/* Clean expander */
.streamlit-expanderHeader {
    font-weight: 600;
}
</style>
"""


def apply_theme():
    """Inject subtle, clean stylesheet."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
