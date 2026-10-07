"""
app/pages/5_gmail_scanner.py
============================
Live Gmail Inbox Scanner with:
- Account-linked credentials (auto-filled from user profile)
- Step-by-step connection diagnostics
- Manual email paste fallback (works without IMAP)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from dotenv import load_dotenv

from spam_detector.config import cfg
from spam_detector.db import init_db
from spam_detector.auth import update_user_gmail_credentials
from app.components.ui_theme import apply_theme

load_dotenv()

st.set_page_config(page_title="Gmail Scanner — SpamShield", page_icon="📬", layout="wide")
apply_theme()
init_db()


from app.components.auth_ui import require_auth
current_user = require_auth()

st.title("📬 Live Gmail Inbox Scanner")

# ---------------------------------------------------------------------------
# Load Predictor
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading DistilBERT model...")
def load_predictor():
    from spam_detector.predict import Predictor
    return Predictor()

model_dir = cfg.model_dir
if not model_dir.exists():
    st.error(
        f"⚠️ Model not trained yet. Run this first:\n\n"
        "```powershell\n.venv\\Scripts\\python -m spam_detector.train --csv data/raw/spam.csv\n```"
    )
    st.stop()

predictor = load_predictor()

# ---------------------------------------------------------------------------
# Tabs: Live IMAP Scanner | Diagnostics | Manual Paste
# ---------------------------------------------------------------------------
tab_live, tab_diag, tab_manual = st.tabs([
    "📬 Live IMAP Scanner",
    "🔬 Connection Diagnostics",
    "📋 Manual Email Paste",
])

# ===========================================================================
# TAB 1: Live IMAP Scanner
# ===========================================================================
with tab_live:
    st.markdown("Connect your Gmail to scan real incoming emails for spam.")

    with st.expander("ℹ️ How to get your 16-character Google App Password"):
        st.markdown("""
        1. Go to your **[Google Account Security](https://myaccount.google.com/security)** page.
        2. Ensure **2-Step Verification** is **ON**.
        3. Open **[App Passwords](https://myaccount.google.com/apppasswords)**, enter a name like `SpamShield`, click **Create**.
        4. Copy the **16 letters** shown in the yellow box (e.g. `abcd efgh ijkl mnop`).
        5. Paste below — spaces are automatically removed.
        
        > ⚠️ **Google Workspace / school / company accounts**: Your admin must enable App Passwords in the 
        > [Google Workspace Admin Console](https://admin.google.com) under  
        > **Security → Authentication → Allow users to manage their access to less secure apps**.
        """)

    # Connection status
    if current_user.has_gmail_connected:
        st.success(f"🟢 **Account Linked:** `{current_user.gmail_address}` — credentials saved in your profile.")
    else:
        st.warning("⚠️ No Gmail linked yet. Enter credentials below and check **Save to my profile**.")

    col_user, col_pass = st.columns(2)
    with col_user:
        gmail_user = st.text_input(
            "Gmail Address",
            value=current_user.gmail_address or "",
            placeholder="yourname@gmail.com",
        )
    with col_pass:
        app_password = st.text_input(
            "16-Character Google App Password",
            value=current_user.gmail_app_password or "",
            type="password",
            placeholder="abcd efgh ijkl mnop",
        )

    col_folder, col_limit, col_unread, col_save = st.columns([2, 1, 1, 2])
    with col_folder:
        folder = st.selectbox("Mailbox folder", ["INBOX", "[Gmail]/Spam", "[Gmail]/All Mail"], index=0)
    with col_limit:
        limit = st.number_input("Emails", min_value=1, max_value=50, value=10, step=1)
    with col_unread:
        st.write("")
        unread_only = st.checkbox("Unread only", value=False)
    with col_save:
        st.write("")
        save_creds = st.checkbox("Save to my profile", value=True)

    if st.button("🚀 Scan Gmail Inbox", type="primary", disabled=not (gmail_user.strip() and app_password.strip())):
        if save_creds:
            update_user_gmail_credentials(current_user.id, gmail_user.strip(), app_password.strip())
            current_user.gmail_address = gmail_user.strip()
            current_user.gmail_app_password = app_password.replace(" ", "").strip()
            st.session_state["authenticated_user"] = current_user

        from spam_detector.gmail import scan_gmail
        with st.spinner(f"Connecting to Gmail SSL server and scanning {limit} emails..."):
            try:
                results = scan_gmail(
                    predictor=predictor,
                    username=gmail_user.strip(),
                    app_password=app_password.strip(),
                    folder=folder,
                    limit=int(limit),
                    unread_only=unread_only,
                    log_to_db=True,
                )
                st.session_state["gmail_results"] = results
                st.session_state["scanned_account"] = gmail_user.strip()
            except ValueError as e:
                st.error(f"❌ {e}")
                st.info("💡 **Tip:** Use the **🔬 Connection Diagnostics** tab above to identify the exact failure point.")

    # Results
    if "gmail_results" in st.session_state:
        results = st.session_state["gmail_results"]
        if not results:
            st.info("No emails found for the selected criteria.")
        else:
            spam_count = sum(1 for r in results if r["is_spam"])
            ham_count = len(results) - spam_count
            avg_conf = sum(r["confidence"] for r in results) / len(results)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📧 Total Scanned", len(results))
            m2.metric("🚨 SPAM Detected", spam_count)
            m3.metric("✅ Legitimate HAM", ham_count)
            m4.metric("📊 Avg Confidence", f"{avg_conf:.1%}")

            st.subheader("📬 Email Threat Results")
            for idx, item in enumerate(results):
                badge = "🚨 SPAM" if item["is_spam"] else "✅ HAM"
                header = f"{badge}  |  **{item['subject'] or '(No Subject)'}**  —  From: `{item['sender'] or 'Unknown'}`  ({item['confidence']:.1%})"
                with st.expander(header, expanded=item["is_spam"]):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Date:** {item['date']}")
                        st.markdown(f"**Sender:** `{item['sender']}`")
                        st.markdown(f"**Subject:** {item['subject']}")
                        st.text(item["eval_text"][:500] + ("..." if len(item["eval_text"]) > 500 else ""))
                    with c2:
                        st.metric("Label", item["label"])
                        st.metric("Confidence", f"{item['confidence']:.2%}")
                    st.progress(item["spam_prob"], text=f"SPAM: {item['spam_prob']:.1%}")
                    st.progress(item["ham_prob"], text=f"HAM:  {item['ham_prob']:.1%}")


                    if st.button("💡 Explain with SHAP", key=f"shap_{idx}"):
                        from spam_detector.explain import explain_prediction, shap_to_html
                        with st.spinner("Running SHAP analysis..."):
                            tokens = explain_prediction(item["eval_text"], predictor._pipeline, max_evals=250)
                            st.markdown(shap_to_html(tokens), unsafe_allow_html=True)


# ===========================================================================
# TAB 2: Connection Diagnostics
# ===========================================================================
with tab_diag:
    st.subheader("🔬 Step-by-Step Gmail Connection Diagnostics")
    st.markdown(
        "Runs 6 independent checks to identify exactly where authentication fails. "
        "Use this if the scanner gives **Invalid credentials** errors."
    )

    diag_email = st.text_input("Gmail Address to test", value=current_user.gmail_address or "", key="diag_email")
    diag_pass = st.text_input(
        "App Password to test",
        type="password",
        value=current_user.gmail_app_password or "",
        key="diag_pass",
        placeholder="abcd efgh ijkl mnop",
    )

    if st.button("🔬 Run Diagnostics", type="primary"):
        if not diag_email or not diag_pass:
            st.error("Please enter both email and password.")
        else:
            from spam_detector.gmail_diagnostics import diagnose_gmail_connection
            with st.spinner("Running diagnostics..."):
                steps = diagnose_gmail_connection(diag_email.strip(), diag_pass.strip())

            st.markdown("---")
            all_ok = all(s["ok"] for s in steps)
            if all_ok:
                st.success("✅ **All checks passed!** Your Gmail connection is working correctly.")
            else:
                st.error("❌ **Issue detected.** See the step that failed below for guidance.")

            for i, step in enumerate(steps):
                icon = "✅" if step["ok"] else "❌"
                color = "#065f46" if step["ok"] else "#991b1b"
                bg = "#ecfdf5" if step["ok"] else "#fef2f2"
                border = "#a7f3d0" if step["ok"] else "#fecaca"
                st.markdown(
                    f"""
                    <div style="background:{bg}; border:1px solid {border}; border-radius:8px;
                                padding:12px; margin-bottom:8px;">
                        <div style="font-weight:700; color:{color}; font-size:1rem;">
                            Step {i+1}: {icon} {step['step']}
                        </div>
                        <div style="color:{color}; font-size:0.88rem; margin-top:4px; white-space:pre-line;">
                            {step['message']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if not all_ok:
                st.markdown("---")
                st.info(
                    "**If authentication fails on a Google Workspace / school / company account:**\n\n"
                    "Ask your admin to enable App Passwords at:\n"
                    "**Google Admin Console → Security → Authentication → Allow users to manage App Passwords**\n\n"
                    "**Alternative:** Use the **📋 Manual Email Paste** tab — no IMAP required!"
                )

# ===========================================================================
# TAB 3: Manual Email Paste Fallback
# ===========================================================================
with tab_manual:
    st.subheader("📋 Manual Email Paste — No IMAP Required")
    st.markdown(
        "Paste the raw content of any email below to classify it instantly. "
        "This works regardless of your Gmail settings or account type."
    )

    sample_spam = """Subject: Congratulations! You've won a $1,000 Gift Card!

Dear Winner,

You have been selected to receive a $1,000 Walmart Gift Card! Click the link below immediately to claim your prize before it expires:

http://win-giftcard-now-free.xyz/claim?id=283791

Hurry — this offer expires in 24 hours! Reply with your full name, address, and credit card number for verification.

Best regards,
Prize Team"""

    sample_ham = """Subject: Meeting notes from today's standup

Hi team,

Here are the notes from today's standup:

- Alex is working on the deployment pipeline
- Sarah finished the unit tests for the auth module
- We agreed to push the release to Friday

Let me know if I missed anything.

Thanks,
Jordan"""

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("📌 Load Sample SPAM Email", use_container_width=True):
            st.session_state["manual_email_text"] = sample_spam
    with col_s2:
        if st.button("📌 Load Sample HAM Email", use_container_width=True):
            st.session_state["manual_email_text"] = sample_ham

    email_text = st.text_area(
        "Paste full email content here (subject + body)",
        value=st.session_state.get("manual_email_text", ""),
        height=300,
        placeholder="Subject: Your email here...\n\nEmail body...",
    )

    if st.button("🔍 Classify This Email", type="primary", disabled=not email_text.strip()):
        with st.spinner("Classifying..."):
            pred = predictor.predict(email_text)

        from spam_detector.db import log_prediction
        log_prediction(pred, source="manual")

        badge = "🚨 SPAM" if pred.is_spam else "✅ HAM"
        color = "#EF4444" if pred.is_spam else "#22C55E"

        st.markdown(
            f"""
            <div style="background:{color}22; border:2px solid {color}; border-radius:12px;
                        padding:20px; margin:16px 0; text-align:center;">
                <div style="font-size:2rem; font-weight:800; color:{color};">{badge}</div>
                <div style="font-size:1.1rem; margin-top:8px; color:#1e293b;">
                    Confidence: <strong>{pred.confidence:.2%}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        col_a.metric("HAM Probability", f"{pred.ham_prob:.2%}")
        col_b.metric("SPAM Probability", f"{pred.spam_prob:.2%}")

        if st.checkbox("💡 Show SHAP word-level explanation", value=False):
            from spam_detector.explain import explain_prediction, shap_to_html
            with st.spinner("Computing SHAP values..."):
                tokens = explain_prediction(email_text, predictor._pipeline, max_evals=250)
                st.markdown(shap_to_html(tokens), unsafe_allow_html=True)
