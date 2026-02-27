"""
vectorstore.py
--------------
Builds and manages a FAISS vector store using
HuggingFace sentence-transformer embeddings (free, local, no API key).
"""

import os
import pickle
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Model is small (~90MB), cached locally after first download
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_store.pkl")


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Embeds all document chunks and returns a FAISS vector store.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    with open(STORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)


def load_vectorstore() -> Optional[FAISS]:
    if os.path.exists(STORE_PATH):
        with open(STORE_PATH, "rb") as f:
            return pickle.load(f)
    return None
