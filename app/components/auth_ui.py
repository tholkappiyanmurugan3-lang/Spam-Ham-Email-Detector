"""
app/components/auth_ui.py
=========================
Simple, attractive, and reliable authentication interface for SpamShield.
"""

from __future__ import annotations

import streamlit as st
from spam_detector.auth import (
    User,
    authenticate_user,
    register_user,
    init_auth_db,
    update_user_gmail_credentials,
    create_reset_token,
    validate_and_reset_password,
    send_reset_email_smtp,
)
from spam_detector.db import _get_connection
from typing import Optional

init_auth_db()


def render_login_portal() -> None:
    """Render a clean, attractive, centered login portal."""
    _, col, _ = st.columns([1, 2, 1])

    with col:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 1.2rem;">
                    <h2 style="margin: 0; font-weight: 700;">🛡️ SpamShield</h2>
                    <p style="color: gray; margin: 4px 0 0; font-size: 0.9rem;">Email Spam & Threat Detection</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            tab_login, tab_reg, tab_forgot = st.tabs(["Sign In", "Create Account", "Forgot Password"])

            # -------------------------------------------------------------
            # TAB 1: Sign In
            # -------------------------------------------------------------
            with tab_login:
                st.markdown("")
                if st.button("⚡ Fill Demo Admin (`admin` / `admin123`)", use_container_width=True):
                    st.session_state["login_u"] = "admin"
                    st.session_state["login_p"] = "admin123"

                with st.form("form_signin"):
                    u = st.text_input("Username or Email", value=st.session_state.get("login_u", ""), placeholder="admin")
                    p = st.text_input("Password", value=st.session_state.get("login_p", ""), type="password", placeholder="••••••••")
                    sub_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)

                if sub_login:
                    if not u or not p:
                        st.error("Please enter both username and password.")
                    else:
                        user, msg = authenticate_user(u, p)
                        if user:
                            st.session_state["authenticated_user"] = user
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

            # -------------------------------------------------------------
            # TAB 2: Create Account
            # -------------------------------------------------------------
            with tab_reg:
                st.markdown("")
                with st.form("form_signup"):
                    full_name = st.text_input("Full Name", placeholder="Alex Rivera")
                    new_u = st.text_input("Username", placeholder="alex")
                    new_e = st.text_input("Email", placeholder="alex@company.com")
                    new_p = st.text_input("Password (min 6 characters)", type="password", placeholder="••••••••")
                    role = st.selectbox("Role", ["Analyst", "Auditor", "Admin"])

                    with st.expander("📬 Optional: Link Gmail"):
                        gm_user = st.text_input("Gmail Address", placeholder="you@gmail.com")
                        gm_pass = st.text_input("16-char App Password", type="password", placeholder="abcd efgh ijkl mnop")

                    sub_reg = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if sub_reg:
                    ok, msg = register_user(
                        username=new_u,
                        email=new_e,
                        password=new_p,
                        full_name=full_name,
                        role=role.lower(),
                        gmail_address=gm_user or None,
                        gmail_app_password=gm_pass or None,
                    )
                    if ok:
                        st.success(f"✅ {msg}")
                        st.info("Switch to the 'Sign In' tab above to log in.")
                    else:
                        st.error(f"❌ {msg}")

            # -------------------------------------------------------------
            # TAB 3: Forgot Password
            # -------------------------------------------------------------
            with tab_forgot:
                st.markdown("")
                st.caption("Enter your email address to receive or view a 6-digit reset code.")

                with st.form("form_forgot_code"):
                    reset_email = st.text_input("Registered Email", placeholder="you@company.com")
                    sub_code = st.form_submit_button("Request Reset Code", use_container_width=True)

                if sub_code:
                    if not reset_email or "@" not in reset_email:
                        st.error("Please enter a valid email address.")
                    else:
                        ok, msg, token = create_reset_token(reset_email.strip())
                        if not ok:
                            st.info("📬 If that email is registered, a reset code has been generated.")
                        else:
                            st.session_state["reset_for_email"] = reset_email.strip().lower()
                            st.session_state["shown_token"] = token
                            st.success(f"Reset code generated for {reset_email}!")

                if "shown_token" in st.session_state:
                    st.info(f"🔑 Your Reset Code: **`{st.session_state['shown_token']}`** (valid for 15 mins)")

                    with st.form("form_reset_apply"):
                        otp = st.text_input("6-digit Code", max_chars=6, placeholder="123456")
                        npw = st.text_input("New Password", type="password", placeholder="••••••••")
                        cpw = st.text_input("Confirm Password", type="password", placeholder="••••••••")
                        sub_apply = st.form_submit_button("Reset Password", type="primary", use_container_width=True)

                    if sub_apply:
                        if not otp:
                            st.error("Please enter the code.")
                        elif npw != cpw:
                            st.error("Passwords do not match.")
                        elif len(npw) < 6:
                            st.error("Password must be at least 6 characters.")
                        else:
                            ok, msg = validate_and_reset_password(
                                st.session_state.get("reset_for_email", ""),
                                otp.strip(),
                                npw,
                            )
                            if ok:
                                st.success(f"✅ {msg}")
                                st.balloons()
                                st.session_state.pop("shown_token", None)
                                st.session_state.pop("reset_for_email", None)
                            else:
                                st.error(f"❌ {msg}")


def render_user_sidebar(user: User) -> None:
    """Render a clean and simple profile widget in sidebar."""
    with st.sidebar:
        with st.container(border=True):
            st.markdown(f"**👤 {user.full_name}**")
            st.caption(f"@{user.username} • {user.email}")
            st.caption(f"Role: **{user.role.upper()}**")

            if user.has_gmail_connected:
                st.success(f"Gmail: `{user.gmail_address}`", icon="🟢")
            else:
                st.warning("Gmail not connected", icon="⚪")

        with st.expander("⚙️ Manage Gmail Link"):
            g_email = st.text_input("Gmail", value=user.gmail_address or "", key="sb_gm_u")
            g_pass = st.text_input("App Password", type="password", key="sb_gm_p", placeholder="xxxx xxxx xxxx xxxx")
            if st.button("Save Gmail", use_container_width=True):
                if g_email and g_pass:
                    update_user_gmail_credentials(user.id, g_email, g_pass)
                    user.gmail_address = g_email
                    user.gmail_app_password = g_pass.replace(" ", "")
                    st.session_state["authenticated_user"] = user
                    st.success("Saved!")
                    st.rerun()
                else:
                    st.error("Enter both email and App Password.")

        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()


def require_auth() -> User:
    """Page guard. If unauthenticated, show login and stop."""
    user: Optional[User] = st.session_state.get("authenticated_user")
    if not user:
        render_login_portal()
        st.stop()
    render_user_sidebar(user)
    return user
