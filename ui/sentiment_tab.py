"""
sentiment_tab.py
----------------
Sentiment Analysis tab — theme-aware, native Streamlit components.
"""

import streamlit as st
from backend.sentiment import analyze_all_sentiments


SENTIMENT_EMOJI  = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
CONFIDENCE_LABEL = {"High": "High confidence", "Medium": "Medium confidence", "Low": "Low confidence"}


def render_sentiment_tab(groq_api_key: str):
    """Renders the Sentiment Analysis tab."""

    if not st.session_state.get("articles_loaded"):
        st.markdown("<br><br>", unsafe_allow_html=True)
        col = st.columns([1, 2, 1])[1]
        with col:
            st.markdown("### 📈 No Articles Loaded")
            st.markdown("Process articles first to see sentiment analysis here.")
        return

    article_meta = st.session_state.get("article_meta", [])

    if "sentiments" not in st.session_state:
        with st.spinner("🔍 Analysing article sentiment…"):
            st.session_state.sentiments = analyze_all_sentiments(article_meta, groq_api_key)

    sentiments = st.session_state.sentiments

    # Header
    st.markdown("## 📊 Article Sentiment Dashboard")
    st.caption("AI-powered Positive · Negative · Neutral scoring per article")
    st.divider()

    # Summary metrics
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for s in sentiments:
        counts[s.get("sentiment", "Neutral")] += 1

    total = max(len(sentiments), 1)
    c1, c2, c3 = st.columns(3)

    with c1:
        pct = round(counts["Positive"] / total * 100)
        st.metric(
            label="🟢  Positive",
            value=f"{counts['Positive']} articles",
            delta=f"{pct}% of total",
            delta_color="normal",
        )
    with c2:
        pct = round(counts["Negative"] / total * 100)
        st.metric(
            label="🔴  Negative",
            value=f"{counts['Negative']} articles",
            delta=f"{pct}% of total",
            delta_color="inverse",
        )
    with c3:
        pct = round(counts["Neutral"] / total * 100)
        st.metric(
            label="🟡  Neutral",
            value=f"{counts['Neutral']} articles",
            delta=f"{pct}% of total",
            delta_color="off",
        )

    st.divider()

    # Per-article cards
    for i, item in enumerate(sentiments, 1):
        sentiment  = item.get("sentiment", "Neutral")
        confidence = item.get("confidence", "Low")
        reason     = item.get("reason", "")
        signals    = item.get("key_signals", [])
        emoji      = SENTIMENT_EMOJI.get(sentiment, "🟡")
        conf_label = CONFIDENCE_LABEL.get(confidence, confidence)

        with st.container(border=True):
            # Title + badges row
            title_col, badge_col = st.columns([4, 1])
            with title_col:
                st.markdown(
                    f"**{i}. [{item.get('title', f'Article {i}')}]({item.get('url', '#')})**"
                )
            with badge_col:
                st.markdown(f"**{emoji} {sentiment}**")

            # Confidence + reason
            st.caption(f"{conf_label}")
            st.markdown(f"**Analysis:** {reason}")

            # Signal pills as a readable list
            if signals:
                signals_text = " · ".join(signals)
                st.caption(f"📌 Key signals: {signals_text}")

        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔄 Re-Analyse Sentiment"):
        st.session_state.pop("sentiments", None)
        st.rerun()
