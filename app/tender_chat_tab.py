"""
T5-1: Chat with Tender tab — natural language query interface.
"""
import streamlit as st
from typing import Optional


def render_tender_chat_tab(payload: dict, widget_key_fn) -> None:
    """Render the Chat with Tender tab."""
    try:
        from src.analysis.tender_chat import answer_query
    except ImportError as e:
        st.error(f"tender_chat module not available: {e}")
        return

    if not payload:
        st.info("Upload and analyse a tender to start querying.")
        return

    st.markdown("### \U0001f4ac Chat with Tender")
    st.caption("Ask anything about this tender in plain English.")

    # Chat history in session state
    history_key = widget_key_fn("chat_history")
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    # Suggested questions
    st.markdown("**Try asking:**")
    suggestions = [
        "Show me all civil RFIs",
        "What are the blockers?",
        "BOQ items over \u20b910L",
        "Total cost by trade",
        "How many pages were processed?",
        "What is the quality score?",
    ]
    cols = st.columns(3)
    suggestion_clicked = None
    for idx, sug in enumerate(suggestions):
        if cols[idx % 3].button(sug, key=widget_key_fn(f"sug_{idx}"), use_container_width=True):
            suggestion_clicked = sug

    st.markdown("---")

    # Input
    query = st.text_input(
        "Your question",
        placeholder="e.g. 'show high priority RFIs' or 'what is the total MEP cost?'",
        key=widget_key_fn("chat_input"),
        label_visibility="collapsed",
    )
    send = st.button("Ask \u2192", key=widget_key_fn("chat_send"), type="primary")

    # Handle suggestion click or send
    active_query = suggestion_clicked or (query if send else None)
    if active_query:
        with st.spinner("Thinking\u2026"):
            resp = answer_query(active_query, payload)
        st.session_state[history_key].append({"q": active_query, "a": resp["answer"], "count": resp["count"]})

    # Render chat history (newest first)
    history = st.session_state.get(history_key, [])
    for item in reversed(history[-20:]):
        with st.chat_message("user"):
            st.markdown(item["q"])
        with st.chat_message("assistant"):
            st.markdown(item["a"])

    if history:
        if st.button("Clear chat", key=widget_key_fn("chat_clear")):
            st.session_state[history_key] = []
            st.rerun()
