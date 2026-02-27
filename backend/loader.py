"""
loader.py
---------
Handles URL loading, text extraction, and chunking.
Uses newspaper3k for robust article extraction.
"""

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict


def fetch_article_text(url: str) -> Dict[str, str]:
    """
    Fetches article text and title from a URL using newspaper3k,
    with a BeautifulSoup fallback.
    Returns a dict with 'title', 'text', and 'url'.
    """
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        title = article.title or url
        text = article.text
        if text.strip():
            return {"title": title, "text": text, "url": url}
    except Exception:
        pass

    # Fallback: BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string if soup.title else url
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs)
        return {"title": title, "text": text, "url": url}
    except Exception as e:
        return {"title": url, "text": f"[Failed to fetch article: {e}]", "url": url}


def load_and_chunk_urls(urls: List[str], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    """
    Loads articles from all given URLs and returns a flat list of
    LangChain Document chunks with source + title metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", ", ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[Document] = []
    article_meta: List[Dict] = []

    for url in urls:
        data = fetch_article_text(url)
        if not data["text"].strip() or data["text"].startswith("[Failed"):
            continue

        doc = Document(
            page_content=data["text"],
            metadata={"source": data["url"], "title": data["title"]},
        )
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
        article_meta.append(data)

    return all_chunks, article_meta
