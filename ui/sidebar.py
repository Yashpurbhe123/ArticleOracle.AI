"""
sidebar.py
----------
Sidebar UI: dynamic URL inputs, process button, status display.
"""

import streamlit as st


def render_sidebar() -> tuple[list[str], bool]:
    """Renders the sidebar. Returns (urls, process_clicked)."""

    # Brand block
    st.sidebar.markdown("""
<div style="padding:8px 8px 8px;text-align:center;
            border-bottom:1px solid var(--divider);">
    <h2 style="margin:0 0 1px;font-size:1.4rem;font-weight:800;
               color:var(--text-primary);letter-spacing:-0.02em;">
       🔮 ArticleOracle AI
    </h2>
    <p style="margin:0;font-size:0.85rem;color:var(--text-muted);font-weight:500;">
        News Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

    # Section label
    st.sidebar.markdown("**Article URLs**")
    st.sidebar.caption("Add up to 10 news article URLs to analyse.")

    # Init count
    if "url_count" not in st.session_state:
        st.session_state.url_count = 3

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("➕ Add URL", use_container_width=True, key="btn_add"):
            if st.session_state.url_count < 10:
                st.session_state.url_count += 1
    with col2:
        if st.button("➖ Remove", use_container_width=True, key="btn_remove"):
            if st.session_state.url_count > 1:
                st.session_state.url_count -= 1

    count = st.session_state.url_count
    st.sidebar.caption(f"{count} / 10 URLs")

    urls = []
    for i in range(count):
        url = st.sidebar.text_input(
            f"URL {i + 1}",
            key=f"url_{i}",
            placeholder=f"https://example.com/article-{i+1}",
            label_visibility="collapsed",
        )
        if url.strip():
            urls.append(url.strip())

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    process_clicked = st.sidebar.button(
        "🚀  Process Articles",
        use_container_width=True,
        type="primary",
    )

    st.sidebar.divider()

    # Status
    if st.session_state.get("articles_loaded"):
        n = len(st.session_state.get("article_meta", []))
        total_chunks = len(st.session_state.get("chunks", []))
        st.sidebar.success(f"✅ {n} article(s) loaded · {total_chunks} chunks indexed")
    else:
        st.sidebar.info("No articles loaded yet.")

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    if st.sidebar.button("🔄  Reset Everything", use_container_width=True):
        for k in ["articles_loaded", "article_meta", "chunks", "vectorstore",
                  "rag_chain", "chat_history", "summaries", "sentiments"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    return urls, process_clicked
