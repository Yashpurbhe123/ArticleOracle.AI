"""
summarizer.py
-------------
Generates a concise TL;DR summary for each article using ChatGroq.
Uses a simple stuff chain for individual article summaries.
"""

from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from typing import List, Dict


SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are a financial news analyst. Read the following news article and produce:

1. A concise 3-5 sentence summary capturing the key points
2. Three bullet-point key takeaways prefixed with "•"
3. One line on how this might affect investors

Article:
{text}

Summary:"""
)


def summarize_article(text: str, groq_api_key: str) -> str:
    """
    Summarizes a single article text using ChatGroq.
    """
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=512,
    )

    # Truncate article to avoid token overload (first 4000 chars is enough)
    truncated = text[:4000]
    prompt = SUMMARY_PROMPT.format(text=truncated)

    response = llm.invoke(prompt)
    return response.content


def summarize_all_articles(article_meta: List[Dict], groq_api_key: str) -> List[Dict]:
    """
    Summarizes each article in article_meta list.
    article_meta: list of dicts with keys 'title', 'text', 'url'
    Returns list of dicts with 'title', 'url', 'summary'.
    """
    results = []
    for article in article_meta:
        summary = summarize_article(article["text"], groq_api_key)
        results.append({
            "title": article["title"],
            "url": article["url"],
            "summary": summary,
        })
    return results
