from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL

_EMBEDDINGS_INSTANCE: Optional[HuggingFaceEmbeddings] = None


def _build_embeddings(local_files_only: bool) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": local_files_only},
    )


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is not None:
        return _EMBEDDINGS_INSTANCE

    # 先嘗試本地快取，避免執行期因網路不穩定導致初始化失敗。
    try:
        _EMBEDDINGS_INSTANCE = _build_embeddings(local_files_only=True)
        return _EMBEDDINGS_INSTANCE
    except Exception:
        pass

    try:
        _EMBEDDINGS_INSTANCE = _build_embeddings(local_files_only=False)
        return _EMBEDDINGS_INSTANCE
    except Exception as exc:
        raise RuntimeError(
            "無法載入 Embedding 模型。請先確認可連網下載 "
            f"`{EMBEDDING_MODEL}`，或先在本機快取該模型後再重試。"
        ) from exc


def create_vector_store(documents: list[Document]) -> Chroma:
    """Create and persist a new Chroma vector store for given documents."""
    if not documents:
        raise ValueError("documents 不可為空，無法建立向量資料庫。")

    persist_dir = Path(CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    collection_name = f"pdf_qa_{uuid4().hex}"
    try:
        return Chroma.from_documents(
            documents=documents,
            embedding=_get_embeddings(),
            persist_directory=str(persist_dir),
            collection_name=collection_name,
        )
    except Exception as exc:
        # Fallback for environments where sqlite file lock / I/O causes persistence failures.
        if "disk i/o error" in str(exc).lower():
            return Chroma.from_documents(
                documents=documents,
                embedding=_get_embeddings(),
                collection_name=collection_name,
            )
        raise


def get_retriever(vector_store: Chroma, top_k: int = 4) -> VectorStoreRetriever:
    """Return a retriever from an existing Chroma vector store."""
    return vector_store.as_retriever(search_kwargs={"k": top_k})
