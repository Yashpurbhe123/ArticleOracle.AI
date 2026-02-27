"""
rag_chain.py
------------
Builds a ConversationalRetrievalChain using:
- ChatGroq (LLaMA 3.1 8B via Groq API)
- FAISS retriever
- ConversationBufferWindowMemory (keeps last 6 exchanges)
"""

import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate


SYSTEM_PROMPT = """You are ArticleOracle AI, an expert news intelligence assistant.
Your job is to answer questions based ONLY on the provided news articles.
Always be factual, concise, and insightful.
If the answer is not in the articles, say so clearly.
When referencing facts, mention the article source.
"""


def build_rag_chain(vectorstore: FAISS, groq_api_key: str) -> ConversationalRetrievalChain:
    """
    Builds and returns a ConversationalRetrievalChain with memory.
    """
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 25},
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=6,  # remember last 6 turns
    )

    # Custom QA prompt that injects the system persona
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""{SYSTEM_PROMPT}

Context from articles:
{{context}}

Question: {{question}}

Answer:"""
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        output_key="answer",
    )

    return chain


def run_query(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Runs a query through the RAG chain.
    Returns dict with 'answer' and 'source_documents'.
    """
    result = chain.invoke({"question": question})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("source_documents", []),
    }
