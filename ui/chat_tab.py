"""
chat_tab.py
-----------
Chat interface tab: message bubbles, source citations, chat history.
"""

import streamlit as st
from backend.rag_chain import run_query


def render_chat_tab():
    """Renders the main chat Q&A interface."""

    if not st.session_state.get("articles_loaded"):
        st.markdown("<br><br>", unsafe_allow_html=True)
        col = st.columns([1, 2, 1])[1]
        with col:
            st.markdown("### 📰 No Articles Loaded")
            st.markdown(
                "Add article URLs in the sidebar and click **🚀 Process Articles** "
                "to start your research session."
            )
        return

    # Status bar — native Streamlit
    article_count = len(st.session_state.get("article_meta", []))
    st.info(
        f"🟢 Knowledge base active · **{article_count}** article(s) indexed · "
        f"**LLaMA 3.1 8B** via Groq",
        icon=None,
    )

    # Init chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing messages
    for msg in st.session_state.chat_history:
        avatar = "🧑‍💼" if msg["role"] == "user" else "🔮"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # Input
    user_input = st.chat_input("Ask anything about the loaded articles…")

    if user_input:
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="🔮"):
            with st.spinner("Thinking…"):
                try:
                    chain = st.session_state.get("rag_chain")
                    result = run_query(chain, user_input)
                    answer = result["answer"]
                    sources = result["sources"]
                    st.markdown(answer)
                    _render_sources(sources)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    err_msg = f"🚨 Error: {e}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": err_msg,
                        "sources": [],
                    })

    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            if "rag_chain" in st.session_state:
                st.session_state.rag_chain.memory.clear()
            st.rerun()


def _render_sources(sources: list):
    """Renders collapsible source citations."""
    if not sources:
        return

    seen = {}
    for doc in sources:
        url   = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", url)
        if url not in seen:
            seen[url] = {"title": title, "snippets": []}
        seen[url]["snippets"].append(doc.page_content[:220])

    with st.expander(f"📌 {len(seen)} source(s) cited", expanded=False):
        for i, (url, data) in enumerate(seen.items()):
            if i > 0:
                st.divider()
            st.markdown(f"**[🔗 {data['title']}]({url})**")
            for snippet in data["snippets"][:2]:
                st.markdown(f"> *…{snippet.strip()}…*")
