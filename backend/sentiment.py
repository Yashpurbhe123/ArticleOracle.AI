"""
sentiment.py
------------
Analyzes the sentiment (Positive / Negative / Neutral) of each article
using a structured LLM prompt through ChatGroq.
"""

from langchain_groq import ChatGroq
from typing import List, Dict
import json
import re


SENTIMENT_PROMPT = """You are a news sentiment analyst. Analyze the following news article and respond ONLY with a valid JSON object in this exact format:

{{
  "sentiment": "Positive" | "Negative" | "Neutral",
  "confidence": "High" | "Medium" | "Low",
  "reason": "One sentence explaining the sentiment",
  "key_signals": ["signal 1", "signal 2", "signal 3"]
}}

Article:
{text}

JSON Response:"""


def analyze_sentiment(text: str, groq_api_key: str) -> Dict:
    """
    Returns a dict with sentiment, confidence, reason, and key_signals.
    Falls back to Neutral on parse errors.
    """
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=300,
    )

    truncated = text[:3000]
    prompt = SENTIMENT_PROMPT.format(text=truncated)

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Extract JSON block if wrapped in markdown
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    return {
        "sentiment": "Neutral",
        "confidence": "Low",
        "reason": "Could not determine sentiment from article.",
        "key_signals": [],
    }


def analyze_all_sentiments(article_meta: List[Dict], groq_api_key: str) -> List[Dict]:
    """
    Runs sentiment analysis on each article.
    Returns list of dicts with 'title', 'url', and sentiment fields.
    """
    results = []
    for article in article_meta:
        sentiment_data = analyze_sentiment(article["text"], groq_api_key)
        results.append({
            "title": article["title"],
            "url": article["url"],
            **sentiment_data,
        })
    return results
