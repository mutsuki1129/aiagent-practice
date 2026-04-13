from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(documents: list[Document]) -> Chroma:
    """Create and persist a new Chroma vector store for given documents."""
    if not documents:
        raise ValueError("documents 不可為空，無法建立向量資料庫。")

    persist_dir = Path(CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=_get_embeddings(),
        persist_directory=str(persist_dir),
        collection_name=f"pdf_qa_{uuid4().hex}",
    )
    return vector_store


def get_retriever(vector_store: Chroma, top_k: int = 4) -> VectorStoreRetriever:
    """Return a retriever from an existing Chroma vector store."""
    return vector_store.as_retriever(search_kwargs={"k": top_k})
