"""
Streamlit login UI component — renders login form and manages session state.

Usage in demo_page.py:
    from src.auth.login_ui import render_auth_gate, get_current_user

    user = render_auth_gate()
    if user:
        # rest of the app
        ...
"""
import streamlit as st
from typing import Optional, Dict
from .supabase_client import is_configured, sign_in, sign_out, get_user


def get_current_user() -> Optional[Dict]:
    """Return current user from session state, or None."""
    token = st.session_state.get("auth_token")
    if not token:
        return None
    # Validate on each page load
    user = get_user(token)
    if not user:
        # Token expired — clear session
        st.session_state.pop("auth_token", None)
        st.session_state.pop("auth_user", None)
    return user


def render_auth_gate() -> Optional[Dict]:
    """
    If Supabase is not configured, return a guest user and skip login.
    If configured, show login form and return user only after auth.
    """
    # No Supabase — run in open/guest mode
    if not is_configured():
        return {"id": "guest", "email": "guest@local", "org_id": "local"}

    user = get_current_user()
    if user:
        return user

    # Show login form
    _render_login_form()
    st.stop()  # halt page rendering until logged in
    return None


def _render_login_form():
    """Render a centered login form."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## xboq.ai")
        st.markdown("#### Sign in to continue")
        with st.form("login_form"):
            email    = st.text_input("Email", placeholder="you@firm.com")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Sign In", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("Enter email and password")
            else:
                with st.spinner("Signing in..."):
                    result = sign_in(email, password)
                if result["error"]:
                    st.error(f"Login failed: {result['error']}")
                else:
                    st.session_state["auth_token"] = result["session"].access_token
                    st.session_state["auth_user"]  = result["user"]
                    st.rerun()


def render_user_menu():
    """Render user info + sign-out button in sidebar."""
    user = get_current_user()
    if not user or user.get("id") == "guest":
        return
    with st.sidebar:
        st.markdown("---")
        st.caption(f"Signed in: {user.get('email', '')}")
        if st.button("Sign Out", use_container_width=True):
            token = st.session_state.pop("auth_token", None)
            st.session_state.pop("auth_user", None)
            if token:
                sign_out(token)
            st.rerun()
