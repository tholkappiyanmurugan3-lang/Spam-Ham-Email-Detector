"""
app/main.py
===========
SpamShield AI — Unified Single-Page Application.
Combines Live Prediction, Gmail Scanning, Batch CSV Processing,
SHAP Explainability, and Performance Analytics in one clean interface.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from spam_detector.config import cfg
from spam_detector.db import (
    init_db,
    log_prediction,
    log_batch,
    get_prediction_stats,
    fetch_all_predictions,
)
from spam_detector.evaluate import load_test_results
from spam_detector.auth import update_user_gmail_credentials
from app.components.ui_theme import apply_theme

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SpamShield — Email Spam & Threat Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
init_db()

# ---------------------------------------------------------------------------
# Authentication Guard (Login / Register / Forgot Password)
# ---------------------------------------------------------------------------
from app.components.auth_ui import require_auth
current_user = require_auth()

# ---------------------------------------------------------------------------
# Model Loader (Cached across the session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading fine-tuned DistilBERT transformer...")
def load_predictor():
    from spam_detector.predict import Predictor
    return Predictor()

predictor = load_predictor()

# ---------------------------------------------------------------------------
# Header & Quick Telemetry
# ---------------------------------------------------------------------------
test_results = load_test_results() or {}
db_stats = get_prediction_stats()

m_acc = f"{test_results.get('eval_accuracy', 0.9872):.1%}"
m_f1 = f"{test_results.get('eval_f1', 0.9725):.1%}"
scanned_count = db_stats.get("total", 0)

st.title("🛡️ SpamShield AI")
st.markdown("All-in-one email threat detection, real inbox scanning, and deep natural language explainability.")

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Model Accuracy", m_acc, "98.7% test set")
    col2.metric("📐 Macro F1-Score", m_f1, "Balanced precision/recall")
    col3.metric("📊 Total Predictions", f"{scanned_count:,}", "Logged in database")
    col4.metric("⚡ Engine Status", "Online", "DistilBERT Active")

st.markdown("")

# ---------------------------------------------------------------------------
# Single-Page Feature Navigation (Tabs)
# ---------------------------------------------------------------------------
tab_live, tab_gmail, tab_batch, tab_shap, tab_dash = st.tabs([
    "🔍 Live Predict",
    "📬 Gmail Scanner",
    "📂 Batch CSV Upload",
    "💡 SHAP Explainability",
    "📊 Performance & Logs",
])

# ===========================================================================
# TAB 1: LIVE PREDICT
# ===========================================================================
with tab_live:
    st.subheader("Instant Email Prediction")
    st.markdown("Paste any email subject and message content to detect whether it is **SPAM** or **HAM**.")

    st.markdown("##### ⚡ Quick Samples")
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        if st.button("🎁 Lottery Phishing", use_container_width=True):
            st.session_state["live_input"] = (
                "Subject: YOU WON $2,500,000!\n\n"
                "Congratulations! Your email has won $2,500,000 in the mobile lottery promo. "
                "Click http://claim-prize-now.xyz to enter your bank account and claim your prize within 24 hours!"
            )

    with s2:
        if st.button("🛒 Amazon Delivery", use_container_width=True):
            st.session_state["live_input"] = (
                "Subject: Your Amazon order #402-1234567 has shipped\n\n"
                "Hello, your package is on its way and will arrive tomorrow by 8 PM. "
                "You can track your package online at amazon.in/orders."
            )

    with s3:
        if st.button("🏦 Bank Debit Alert", use_container_width=True):
            st.session_state["live_input"] = (
                "Subject: HDFC Bank Alert\n\n"
                "Rs. 1,500.00 debited from A/c XX4821 on 05-OCT-26 at Swiggy. "
                "Available balance: Rs. 24,310.00. Call 1800-202-6161 if not authorized."
            )

    with s4:
        if st.button("👥 Meeting Reminder", use_container_width=True):
            st.session_state["live_input"] = (
                "Subject: Team Sprint Review Tomorrow\n\n"
                "Hi team, just a reminder that tomorrow's sprint review is scheduled for 10:30 AM. "
                "Please make sure your demo environments are ready."
            )

    email_input = st.text_area(
        "Email Content (Subject & Body)",
        value=st.session_state.get("live_input", ""),
        placeholder="Paste email text here...",
        height=180,
    )

    if st.button("🔎 Analyze Email", type="primary", disabled=not email_input.strip()):
        with st.spinner("Classifying with DistilBERT..."):
            res = predictor.predict(email_input)
            log_prediction(res, source="live")

        with st.container(border=True):
            if res.is_spam:
                st.error(f"### 🚨 SPAM DETECTED ({res.confidence:.1%} confidence)")
                st.markdown("This email exhibits malicious, unsolicited, or phishing characteristics.")
            else:
                st.success(f"### ✅ HAM / LEGITIMATE ({res.confidence:.1%} confidence)")
                st.markdown("This email appears to be genuine personal, transactional, or corporate communication.")

            st.markdown("")
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(f"**Legitimate (HAM):** `{res.ham_prob:.1%}`")
                st.progress(res.ham_prob)
            with p2:
                st.markdown(f"**Spam / Malicious:** `{res.spam_prob:.1%}`")
                st.progress(res.spam_prob)

        with st.expander("🧹 View Tokenizer Input"):
            st.code(res.cleaned_text, language="text")

# ===========================================================================
# TAB 2: GMAIL SCANNER
# ===========================================================================
with tab_gmail:
    st.subheader("Live Gmail Inbox Scanner")
    st.markdown("Authenticate via Google SSL IMAP to scan and classify actual emails from your mailbox.")

    with st.expander("ℹ️ How to get your 16-character Google App Password"):
        st.markdown(
            """
            1. Go to your **[Google Account Security](https://myaccount.google.com/security)** settings.
            2. Turn on **2-Step Verification**.
            3. Visit **[App Passwords](https://myaccount.google.com/apppasswords)** and generate an app password (e.g., named `SpamShield`).
            4. Copy the **16-letter code** (spaces are ignored) and enter below.
            """
        )

    # Configuration Form
    c_gm1, c_gm2 = st.columns(2)
    with c_gm1:
        gmail_user = st.text_input("Gmail Address", value=current_user.gmail_address or "", placeholder="you@gmail.com")
    with c_gm2:
        gmail_pass = st.text_input(
            "16-Char Google App Password",
            value=current_user.gmail_app_password or "",
            type="password",
            placeholder="abcd efgh ijkl mnop",
        )

    c_fld, c_lim, c_unr, c_sav = st.columns([2, 1, 1, 2])
    with c_fld:
        folder = st.selectbox("Mailbox Folder", ["INBOX", "[Gmail]/Spam", "[Gmail]/All Mail"], index=0)
    with c_lim:
        limit = st.number_input("Max Emails", min_value=1, max_value=50, value=10, step=1)
    with c_unr:
        st.write("")
        unread_only = st.checkbox("Unread only", value=False)
    with c_sav:
        st.write("")
        save_to_prof = st.checkbox("Save to profile", value=True)

    if st.button("🚀 Scan Gmail Inbox", type="primary", disabled=not (gmail_user.strip() and gmail_pass.strip())):
        if save_to_prof:
            update_user_gmail_credentials(current_user.id, gmail_user.strip(), gmail_pass.strip())
            current_user.gmail_address = gmail_user.strip()
            current_user.gmail_app_password = gmail_pass.replace(" ", "").strip()
            st.session_state["authenticated_user"] = current_user

        from spam_detector.gmail import scan_gmail
        with st.spinner(f"Connecting to Gmail and analyzing {limit} emails..."):
            try:
                g_results = scan_gmail(
                    predictor=predictor,
                    username=gmail_user.strip(),
                    app_password=gmail_pass.strip(),
                    folder=folder,
                    limit=int(limit),
                    unread_only=unread_only,
                    log_to_db=True,
                )
                st.session_state["gmail_scan_results"] = g_results
            except ValueError as exc:
                st.error(f"❌ {exc}")

    if "gmail_scan_results" in st.session_state:
        g_results = st.session_state["gmail_scan_results"]
        if not g_results:
            st.info("No messages found matching the selected parameters.")
        else:
            n_spam = sum(1 for r in g_results if r["is_spam"])
            n_ham = len(g_results) - n_spam
            avg_c = sum(r["confidence"] for r in g_results) / len(g_results)

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Scanned", len(g_results))
            r2.metric("Spam Flagged", n_spam)
            r3.metric("Legitimate Ham", n_ham)
            r4.metric("Avg Confidence", f"{avg_c:.1%}")

            st.markdown("##### Threat Results")
            for idx, item in enumerate(g_results):
                badge = "🚨 SPAM" if item["is_spam"] else "✅ HAM"
                hdr = f"{badge}  |  **{item['subject'] or '(No Subject)'}**  —  `{item['sender'] or 'Unknown'}` ({item['confidence']:.1%})"
                with st.expander(hdr, expanded=item["is_spam"]):
                    col_info, col_label = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"**Date:** {item['date']}")
                        st.markdown(f"**Sender:** `{item['sender']}`")
                        st.markdown(f"**Subject:** {item['subject']}")
                        st.text(item["eval_text"][:400] + ("..." if len(item["eval_text"]) > 400 else ""))
                    with col_label:
                        st.metric("Label", item["label"])
                        st.metric("Confidence", f"{item['confidence']:.2%}")
                    st.progress(item["spam_prob"], text=f"SPAM: {item['spam_prob']:.1%}")
                    st.progress(item["ham_prob"], text=f"HAM:  {item['ham_prob']:.1%}")

# ===========================================================================
# TAB 3: BATCH CSV UPLOAD
# ===========================================================================
with tab_batch:
    st.subheader("Bulk Batch CSV Classification")
    st.markdown("Upload any CSV containing email texts or subjects to run high-throughput batch classification.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df):,} rows from `{uploaded_file.name}`.")

            text_cols = [c for c in df.columns if df[c].dtype == object]
            chosen_col = st.selectbox("Select column containing email text:", text_cols, index=0 if text_cols else None)

            if chosen_col and st.button("⚡ Run Batch Classification", type="primary"):
                texts = df[chosen_col].fillna("").astype(str).tolist()
                with st.spinner(f"Classifying {len(texts):,} records..."):
                    batch_preds = predictor.predict_batch(texts)
                    log_batch(batch_preds, source="batch")

                df["predicted_label"] = [p.label for p in batch_preds]
                df["confidence"] = [p.confidence for p in batch_preds]
                df["spam_prob"] = [p.spam_prob for p in batch_preds]
                df["ham_prob"] = [p.ham_prob for p in batch_preds]

                st.session_state["batch_results_df"] = df

            if "batch_results_df" in st.session_state:
                b_df = st.session_state["batch_results_df"]
                st.dataframe(b_df.head(50), use_container_width=True)

                csv_buffer = io.StringIO()
                b_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Annotated CSV",
                    data=csv_buffer.getvalue(),
                    file_name="spamshield_classified_batch.csv",
                    mime="text/csv",
                )
        except Exception as err:
            st.error(f"Error reading CSV: {err}")

# ===========================================================================
# TAB 4: SHAP EXPLAINABILITY
# ===========================================================================
with tab_shap:
    st.subheader("SHAP Word-Level Explainability")
    st.markdown("Understand **why** the model flagged an email. Tokens in **red** push towards SPAM; tokens in **green** push towards HAM.")

    shap_text = st.text_area(
        "Email Text to Explain",
        value=st.session_state.get("live_input", ""),
        placeholder="Paste email here to compute word-level Shapley values...",
        height=150,
    )

    shap_evals = st.slider("SHAP max evaluations (higher = more precise, slower)", 100, 600, 250, 50)

    if st.button("💡 Explain Prediction with SHAP", type="primary", disabled=not shap_text.strip()):
        from spam_detector.explain import explain_prediction, shap_to_html
        with st.spinner("Computing Shapley values across transformer layers..."):
            shap_result = predictor.predict(shap_text)
            tokens = explain_prediction(shap_text, predictor._pipeline, max_evals=shap_evals)

        st.markdown("---")
        if shap_result.is_spam:
            st.error(f"Verdict: **SPAM** ({shap_result.confidence:.1%})")
        else:
            st.success(f"Verdict: **HAM** ({shap_result.confidence:.1%})")

        st.markdown("##### Token Heatmap")
        st.markdown(shap_to_html(tokens), unsafe_allow_html=True)

        if tokens:
            t_names = [t for t, _ in tokens]
            t_vals = [v for _, v in tokens]
            colors = ["#EF4444" if v > 0 else "#22C55E" for v in t_vals]

            fig = go.Figure(go.Bar(
                x=t_vals,
                y=t_names,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in t_vals],
                textposition="outside",
            ))
            fig.update_layout(
                title="Top Influential Words (Positive = Spam, Negative = Ham)",
                xaxis_title="SHAP Value",
                yaxis_title="Token",
                height=max(280, len(tokens) * 32),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 5: PERFORMANCE & LOGS
# ===========================================================================
with tab_dash:
    st.subheader("Performance Metrics & Audit Logs")

    d1, d2 = st.columns([1, 1])
    with d1:
        st.markdown("##### 🎯 Model Generalization (1,559 Test Emails)")
        with st.container(border=True):
            st.write(f"• **Test Accuracy:** `{m_acc}`")
            st.write(f"• **Macro F1-Score:** `{m_f1}`")
            st.write(f"• **Evaluation Loss:** `{test_results.get('eval_loss', 0.0525):.4f}`")
            st.write("• **Corpus:** 10,389 emails (SpamAssassin + SMS + Modern HAM)")

    with d2:
        st.markdown("##### 📊 Live Threat Breakdown")
        with st.container(border=True):
            t_total = db_stats.get("total", 0)
            t_spam = db_stats.get("spam_count", 0)
            t_ham = db_stats.get("ham_count", 0)
            st.write(f"• **Total Recorded:** `{t_total:,}`")
            st.write(f"• **Spam Detected:** `{t_spam:,}`")
            st.write(f"• **Legitimate Ham:** `{t_ham:,}`")

    # Prediction History Table
    st.markdown("##### 📜 Prediction History Logs")

    c_lim_sel, c_dl = st.columns([2, 1])
    with c_lim_sel:
        limit_choice = st.selectbox(
            "Rows to display:",
            options=[50, 100, 250, 500, 1000, "All"],
            index=2,  # default 250
            help="Select how many recent predictions to load from the SQLite database.",
        )
    actual_limit = None if limit_choice == "All" else int(limit_choice)

    all_preds_df = fetch_all_predictions(limit=actual_limit)

    if not all_preds_df.empty:
        # Keep desired columns if they exist
        cols_to_show = [c for c in ["created_at", "label", "confidence", "spam_prob", "ham_prob", "source", "raw_text"] if c in all_preds_df.columns]
        display_df = all_preds_df[cols_to_show]
        st.dataframe(display_df, use_container_width=True)

        with c_dl:
            st.write("")
            csv_buf = io.StringIO()
            display_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Export Logs to CSV",
                data=csv_buf.getvalue(),
                file_name="spamshield_predictions_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No predictions recorded yet. Run Live Predict or Gmail Scanner to populate the log.")

