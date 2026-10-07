"""
app/pages/1_live_predict.py
===========================
Simple, attractive live email prediction page.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from spam_detector.config import cfg
from spam_detector.db import log_prediction, init_db
from app.components.ui_theme import apply_theme

st.set_page_config(page_title="Live Predict — SpamShield", page_icon="🔍", layout="wide")
apply_theme()
init_db()

from app.components.auth_ui import require_auth
current_user = require_auth()

st.title("🔍 Live Email Predictor")
st.markdown("Paste an email subject and body below to check whether it is legitimate (**HAM**) or **SPAM**.")

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    from spam_detector.predict import Predictor
    return Predictor()

predictor = load_predictor()

# ---------------------------------------------------------------------------
# Quick Samples
# ---------------------------------------------------------------------------
st.markdown("##### Quick Test Samples")
s1, s2, s3, s4 = st.columns(4)

with s1:
    if st.button("🎁 Lottery Phishing", use_container_width=True):
        st.session_state["email_input_val"] = (
            "Subject: YOU WON $2,500,000!\n\n"
            "Congratulations! Your email has won $2,500,000 in the mobile lottery promo. "
            "Click http://claim-prize-now.xyz to enter your bank account and claim your prize within 24 hours!"
        )

with s2:
    if st.button("🛒 Amazon Order", use_container_width=True):
        st.session_state["email_input_val"] = (
            "Subject: Your Amazon order #402-1234567 has shipped\n\n"
            "Hello, your package is on its way and will arrive tomorrow by 8 PM. "
            "You can track your package online at amazon.in/orders."
        )

with s3:
    if st.button("🏦 Bank Debit Alert", use_container_width=True):
        st.session_state["email_input_val"] = (
            "Subject: HDFC Bank Alert\n\n"
            "Rs. 1,500.00 debited from A/c XX4821 on 05-OCT-26 at Swiggy. "
            "Available balance: Rs. 24,310.00. Call 1800-202-6161 if not authorized."
        )

with s4:
    if st.button("👥 Meeting Reminder", use_container_width=True):
        st.session_state["email_input_val"] = (
            "Subject: Team Sprint Review Tomorrow\n\n"
            "Hi team, just a reminder that tomorrow's sprint review is scheduled for 10:30 AM. "
            "Please make sure your demo environments are ready."
        )

# Text Area Input
email_input = st.text_area(
    "Email Content",
    value=st.session_state.get("email_input_val", ""),
    placeholder="Paste subject and email body text here...",
    height=200,
)

if st.button("🔎 Analyze Email", type="primary", disabled=not email_input.strip()):
    with st.spinner("Classifying..."):
        result = predictor.predict(email_input)
        log_prediction(result, source="live")

    st.markdown("---")

    # Clean Verdict Card
    with st.container(border=True):
        if result.is_spam:
            st.error(f"### 🚨 SPAM DETECTED ({result.confidence:.1%} confidence)")
            st.markdown("This email looks like unsolicited advertising, phishing, or a scam.")
        else:
            st.success(f"### ✅ HAM / LEGITIMATE ({result.confidence:.1%} confidence)")
            st.markdown("This email appears to be genuine personal, transactional, or work communication.")

        st.markdown("")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**Legitimate (HAM):** `{result.ham_prob:.1%}`")
            st.progress(result.ham_prob)
        with col_m2:
            st.markdown(f"**Spam / Malicious:** `{result.spam_prob:.1%}`")
            st.progress(result.spam_prob)

    with st.expander("🧹 View Preprocessed Text"):
        st.code(result.cleaned_text, language="text")
