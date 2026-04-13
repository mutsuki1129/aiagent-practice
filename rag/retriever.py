from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


def retrieve_documents(retriever: VectorStoreRetriever, query: str) -> list[Document]:
    """Retrieve relevant document chunks for a user query."""
    if not query.strip():
        return []
    return retriever.invoke(query)
