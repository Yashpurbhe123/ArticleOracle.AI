"""
app.py
------
ArticleOracle AI — LLM-Powered News Intelligence Platform
Main Streamlit entry point.

Run with:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from backend.loader import load_and_chunk_urls
from backend.vectorstore import build_vectorstore
from backend.rag_chain import build_rag_chain
from ui.sidebar import render_sidebar
from ui.chat_tab import render_chat_tab
from ui.summary_tab import render_summary_tab
from ui.sentiment_tab import render_sentiment_tab

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ArticleOracle AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS — light + dark theme aware
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── CSS variables: dark mode (default) ──────── */
:root {
    --accent:        #4F8CF7;
    --accent-soft:   rgba(79,140,247,0.12);
    --accent-border: rgba(79,140,247,0.3);
    --surface:       rgba(255,255,255,0.04);
    --surface-hover: rgba(255,255,255,0.07);
    --border:        rgba(255,255,255,0.1);
    --border-strong: rgba(255,255,255,0.18);
    --text-primary:  #f0f2f6;
    --text-secondary:#a8adb8;
    --text-muted:    #62676e;
    --bg-app:        #0e1117;
    --bg-card:       rgba(255,255,255,0.03);
    --bg-sidebar:    #0a0c12;
    --divider:       rgba(255,255,255,0.08);

    --green:        #3dd68c;
    --green-soft:   rgba(61,214,140,0.12);
    --green-border: rgba(61,214,140,0.28);
    --red:          #f06969;
    --red-soft:     rgba(240,105,105,0.1);
    --red-border:   rgba(240,105,105,0.28);
    --yellow:       #e8b84b;
    --yellow-soft:  rgba(232,184,75,0.1);
    --yellow-border:rgba(232,184,75,0.28);
}

/* ── CSS variables: light mode ───────────────── */
@media (prefers-color-scheme: light) {
    :root {
        --accent:        #2563eb;
        --accent-soft:   rgba(37,99,235,0.08);
        --accent-border: rgba(37,99,235,0.25);
        --surface:       rgba(0,0,0,0.035);
        --surface-hover: rgba(0,0,0,0.06);
        --border:        rgba(0,0,0,0.1);
        --border-strong: rgba(0,0,0,0.2);
        --text-primary:  #111827;
        --text-secondary:#374151;
        --text-muted:    #6b7280;
        --bg-app:        #f8fafc;
        --bg-card:       rgba(0,0,0,0.02);
        --bg-sidebar:    #f1f5f9;
        --divider:       rgba(0,0,0,0.08);

        --green:        #16a34a;
        --green-soft:   rgba(22,163,74,0.08);
        --green-border: rgba(22,163,74,0.25);
        --red:          #dc2626;
        --red-soft:     rgba(220,38,38,0.08);
        --red-border:   rgba(220,38,38,0.25);
        --yellow:       #ca8a04;
        --yellow-soft:  rgba(202,138,4,0.08);
        --yellow-border:rgba(202,138,4,0.25);
    }
}

/* Also respect Streamlit's explicit theme attribute */
[data-theme="light"] {
    --accent:        #2563eb !important;
    --accent-soft:   rgba(37,99,235,0.08) !important;
    --accent-border: rgba(37,99,235,0.25) !important;
    --surface:       rgba(0,0,0,0.035) !important;
    --surface-hover: rgba(0,0,0,0.06) !important;
    --border:        rgba(0,0,0,0.1) !important;
    --border-strong: rgba(0,0,0,0.2) !important;
    --text-primary:  #111827 !important;
    --text-secondary:#374151 !important;
    --text-muted:    #6b7280 !important;
    --divider:       rgba(0,0,0,0.08) !important;
}

/* ── Base font ───────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-size: 18px !important;
}

/* ── Hide Streamlit chrome ───────────────────── */
footer { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Padding ─────────────────────────────────── */
.main .block-container {
    padding-top: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1100px !important;
}

/* ── Body text ─────────────────────────────────── */
p, li {
    font-size: 1.05rem !important;
    line-height: 1.65 !important;
}

/* ── Headings ─────────────────────────────────── */
h1 { font-size: 2.4rem !important; font-weight: 800 !important; color: var(--text-primary) !important; }
h2 { font-size: 1.75rem !important; font-weight: 700 !important; color: var(--text-primary) !important; }
h3 { font-size: 1.35rem !important; font-weight: 700 !important; color: var(--text-primary) !important; }
h4 { font-size: 1.1rem  !important; font-weight: 700 !important; color: var(--text-primary) !important; }

/* ── Sidebar ─────────────────────────────────── */
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--divider) !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}
/* ── Sidebar text (caption/labels only, NOT brand block) ── */
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stCaption"] p {
    font-size: 0.9rem !important;
}

/* ── Tabs ─────────────────────────────────────── */
div[data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--divider) !important;
    border-radius: 10px 10px 0 0 !important;
    gap: 4px !important;
    padding: 6px 6px 0 !important;
    justify-content: center !important;   /* ← CENTER TABS */
    display: flex !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 12px 28px !important;
    transition: all 0.18s ease !important;
    border-bottom: 2px solid transparent !important;
    white-space: nowrap !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--surface-hover) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--accent-soft) !important;
    border-bottom: 2px solid var(--accent) !important;
}
div[data-baseweb="tab-highlight"] { display: none !important; }

/* ── Primary button ──────────────────────────── */
div.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55em 1.4em !important;
    transition: opacity 0.18s ease, transform 0.14s ease !important;
    box-shadow: 0 2px 12px rgba(79,140,247,0.2) !important;
}
div.stButton > button[kind="primary"]:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary buttons ───────────────────────── */
div.stButton > button:not([kind="primary"]) {
    background: var(--surface) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    transition: all 0.18s ease !important;
}
div.stButton > button:not([kind="primary"]):hover {
    background: var(--surface-hover) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-strong) !important;
}

/* ── Inputs ──────────────────────────────────── */
input[type="text"], textarea {
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

/* ── Chat messages ────────────────────────────── */
div[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    padding: 6px 12px !important;
}

/* ── Chat input ──────────────────────────────── */
div[data-testid="stChatInput"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    font-size: 1rem !important;
    color: var(--text-primary) !important;
}

/* ── Expanders ────────────────────────────────── */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
details summary {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
}

/* ── Progress bar ────────────────────────────── */
div[data-testid="stProgressBar"] > div > div {
    background: var(--accent) !important;
    border-radius: 100px !important;
}

/* ── Caption ─────────────────────────────────── */
small, .stCaption, [data-testid="stCaption"] p {
    font-size: 0.88rem !important;
    color: var(--text-muted) !important;
}

/* ── Scrollbar ────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Divider ─────────────────────────────────── */
hr { border-color: var(--divider) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 18px;text-align:center;">
    <h1 style="font-size:2.2rem;font-weight:800;margin:0 0 8px;
               color:var(--text-primary);letter-spacing:-0.02em;">
        🔮 ArticleOracle AI
    </h1>
    <p style="font-size:1.05rem;color:var(--text-muted);margin:0;">
        RAG &nbsp;&middot;&nbsp; Conversational Memory &nbsp;&middot;&nbsp;
        Sentiment Analysis &nbsp;&middot;&nbsp; Auto-Summarization
    </p>
</div>
<hr style="margin:0 0 10px;">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# API Key Guard
# ─────────────────────────────────────────────
if not GROQ_API_KEY:
    st.error("⚠️ **GROQ_API_KEY not found.** Add it to your `.env` file: `GROQ_API_KEY=your_key_here`")
    st.info("Get your free API key at [console.groq.com](https://console.groq.com)")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
urls, process_clicked = render_sidebar()

# ─────────────────────────────────────────────
# Processing Pipeline
# ─────────────────────────────────────────────
if process_clicked:
    if not urls:
        st.warning("⚠️ Please enter at least one article URL in the sidebar.")
    else:
        for key in ["summaries", "sentiments", "chat_history", "rag_chain"]:
            st.session_state.pop(key, None)

        progress_bar = st.progress(0, text="🔄 Starting pipeline...")

        with st.spinner("Processing articles…"):
            try:
                progress_bar.progress(20, text="📰 Fetching and chunking articles…")
                chunks, article_meta = load_and_chunk_urls(urls)

                if not chunks:
                    progress_bar.empty()
                    st.error("❌ Could not extract text from any URL. Please try different links.")
                    st.stop()

                progress_bar.progress(55, text="🧠 Building FAISS vector store…")
                vectorstore = build_vectorstore(chunks)

                progress_bar.progress(85, text="⛓️ Initializing RAG chain with memory…")
                rag_chain = build_rag_chain(vectorstore, GROQ_API_KEY)

                st.session_state.articles_loaded = True
                st.session_state.article_meta = article_meta
                st.session_state.chunks = chunks
                st.session_state.vectorstore = vectorstore
                st.session_state.rag_chain = rag_chain

                progress_bar.progress(100, text="✅ Done!")
                st.success(
                    f"✅ Processed **{len(article_meta)}** article(s) → "
                    f"**{len(chunks)}** chunks indexed."
                )
                progress_bar.empty()

            except Exception as e:
                progress_bar.empty()
                st.error(f"🚨 Processing failed: {e}")

# ─────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬  Research Chat",
    "📋  Article Summaries",
    "📊  Sentiment Analysis",
    "ℹ️  How To Use",
])

with tab1:
    render_chat_tab()

with tab2:
    render_summary_tab(GROQ_API_KEY)

with tab3:
    render_sentiment_tab(GROQ_API_KEY)

with tab4:
    st.markdown("## ℹ️ How To Use ArticleOracle AI")
    st.markdown("Follow these steps to analyse any set of news articles.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Step 1 — Add Article URLs")
        st.markdown(
            "Paste news article URLs in the left sidebar. "
            "Use **➕ Add URL** to add more (up to 10), **➖ Remove** to remove the last one."
        )
    with col2:
        st.markdown("#### Step 2 — Process Articles")
        st.markdown(
            "Click **🚀 Process Articles**. The system fetches content, splits it into chunks, "
            "embeds them using `all-MiniLM-L6-v2`, and indexes them into FAISS."
        )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Step 3 — Research Chat 💬")
        st.markdown(
            "Head to the **Research Chat** tab. Ask any question — the AI retrieves "
            "relevant chunks, answers with citations, and remembers the last 6 turns."
        )
    with col4:
        st.markdown("#### Step 4 — Summaries & Sentiment 📊")
        st.markdown(
            "The **Article Summaries** tab shows TL;DR + key takeaways. "
            "The **Sentiment Analysis** tab gives a Positive / Negative / Neutral verdict per article."
        )

    st.markdown("---")
    st.markdown("#### 💡 Tips")
    st.markdown(
        "- Works well with **most news and article** URLs\n"
        "- Ask focused questions: *\"What are the main arguments presented in the article?\"*\n"
        "- Use follow-ups — the AI remembers context across turns\n"
        "- Click **🔄 Reset Everything** in the sidebar to start fresh\n"
        "- The HuggingFace embedding model downloads once (~90 MB) and is cached"
    )
    st.markdown("---")
    st.caption("Powered by LLaMA 3.1 8B · Groq · LangChain · FAISS · HuggingFace · Streamlit")
