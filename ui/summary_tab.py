"""
summary_tab.py
--------------
Article Summaries tab — theme-aware, native Streamlit components.
"""

import streamlit as st
from backend.summarizer import summarize_all_articles


def render_summary_tab(groq_api_key: str):
    """Renders the Article Summaries tab."""

    if not st.session_state.get("articles_loaded"):
        st.markdown("<br><br>", unsafe_allow_html=True)
        col = st.columns([1, 2, 1])[1]
        with col:
            st.markdown("### 📋 No Articles Loaded")
            st.markdown("Process articles first to see AI-generated summaries here.")
        return

    article_meta = st.session_state.get("article_meta", [])

    if "summaries" not in st.session_state:
        with st.spinner("📝 Generating AI summaries…"):
            st.session_state.summaries = summarize_all_articles(article_meta, groq_api_key)

    summaries = st.session_state.summaries

    # Header
    col_a, col_b = st.columns([5, 1])
    with col_a:
        st.markdown("## 📋 Article Summaries")
        st.caption("AI-generated TL;DR · Key Takeaways · Investor Insights")
    with col_b:
        st.markdown(f"**{len(summaries)}** article(s)")

    st.divider()

    # Cards
    for i, item in enumerate(summaries, 1):
        with st.container(border=True):
            # Title row
            st.markdown(
                f"**{i}. [{item['title']}]({item['url']})**",
            )
            st.divider()
            # Summary content
            st.markdown(item["summary"])

        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔄 Regenerate Summaries"):
        st.session_state.pop("summaries", None)
        st.rerun()
