from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
import hashlib
import json
import time

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_COLLECTION_PREFIX,
    CHROMA_MAX_COLLECTIONS,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
)

_EMBEDDINGS_INSTANCE: Optional[HuggingFaceEmbeddings] = None
_VECTOR_STORE_CACHE: dict[str, Chroma] = {}
_VECTOR_STORE_CACHE_ORDER: list[str] = []
_VECTOR_STORE_CACHE_LIMIT = 6
_USAGE_FILE_NAME = "collection_usage.json"
_VECTOR_STORE_STATS: dict[str, Any] = {
    "cache_hits": 0,
    "reused_collections": 0,
    "built_collections": 0,
    "cleanup_removed_collections": 0,
    "fallback_builds": 0,
    "last_collection": "",
    "last_fingerprint": "",
}


def _build_embeddings(local_files_only: bool) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": local_files_only},
    )


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is not None:
        return _EMBEDDINGS_INSTANCE

    # Prefer local cache first to avoid runtime network dependency.
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
            "Unable to load embedding model. Ensure network access for initial download "
            f"of `{EMBEDDING_MODEL}` or pre-cache it locally."
        ) from exc


def _doc_fingerprint(documents: list[Document]) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"count={len(documents)}".encode("utf-8"))
    for doc in documents[:5000]:
        source = str(doc.metadata.get("source", "") or "")
        page = str(doc.metadata.get("page", "") or "")
        source_type = str(doc.metadata.get("source_type", "") or "")
        content = str(getattr(doc, "page_content", "") or "")
        head = content[:500]
        hasher.update(source.encode("utf-8", errors="ignore"))
        hasher.update(b"|")
        hasher.update(page.encode("utf-8", errors="ignore"))
        hasher.update(b"|")
        hasher.update(source_type.encode("utf-8", errors="ignore"))
        hasher.update(b"|")
        hasher.update(str(len(content)).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(head.encode("utf-8", errors="ignore"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _collection_name_from_fingerprint(fingerprint: str) -> str:
    prefix = (CHROMA_COLLECTION_PREFIX or "pdf_qa").strip() or "pdf_qa"
    safe_prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in {"_", "-"}).strip("_- ") or "pdf_qa"
    return f"{safe_prefix}_{fingerprint[:16]}"


def _usage_file_path(persist_dir: Path) -> Path:
    return persist_dir / _USAGE_FILE_NAME


def _read_usage_registry(persist_dir: Path) -> dict[str, float]:
    path = _usage_file_path(persist_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            normalized: dict[str, float] = {}
            for key, value in data.items():
                try:
                    normalized[str(key)] = float(value)
                except Exception:
                    continue
            return normalized
    except Exception:
        return {}
    return {}


def _write_usage_registry(persist_dir: Path, usage: dict[str, float]) -> None:
    path = _usage_file_path(persist_dir)
    try:
        path.write_text(json.dumps(usage, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _touch_collection_usage(persist_dir: Path, collection_name: str) -> None:
    usage = _read_usage_registry(persist_dir)
    usage[collection_name] = float(time.time())
    _write_usage_registry(persist_dir, usage)


def _delete_collection(client: Any, collection_name: str) -> None:
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass


def _list_prefixed_collections(client: Any, prefix: str) -> list[str]:
    names: list[str] = []
    try:
        collections = client.list_collections()
    except Exception:
        return names

    expected_prefix = f"{prefix}_"
    for item in collections or []:
        name = ""
        if isinstance(item, str):
            name = item
        else:
            name = str(getattr(item, "name", "") or "")
        if name.startswith(expected_prefix):
            names.append(name)
    return sorted(set(names))


def _cleanup_old_collections(
    client: Any,
    persist_dir: Path,
    *,
    prefix: str,
    max_collections: int,
    keep_collection_name: str,
) -> int:
    limit = max(0, int(max_collections))
    if limit <= 0:
        return 0

    names = _list_prefixed_collections(client, prefix)
    if len(names) <= limit:
        return 0

    usage = _read_usage_registry(persist_dir)
    for name in names:
        usage.setdefault(name, 0.0)

    removal_order = sorted(names, key=lambda name: (usage.get(name, 0.0), name))
    active_names = set(names)
    removed = 0
    for name in removal_order:
        if len(active_names) <= limit:
            break
        if name == keep_collection_name:
            continue
        _delete_collection(client, name)
        active_names.discard(name)
        usage.pop(name, None)
        removed += 1

    for key in list(usage.keys()):
        if key not in active_names:
            usage.pop(key, None)
    _write_usage_registry(persist_dir, usage)
    return removed


def _cache_get(fingerprint: str) -> Chroma | None:
    cached = _VECTOR_STORE_CACHE.get(fingerprint)
    if cached is None:
        return None
    if fingerprint in _VECTOR_STORE_CACHE_ORDER:
        _VECTOR_STORE_CACHE_ORDER.remove(fingerprint)
    _VECTOR_STORE_CACHE_ORDER.append(fingerprint)
    return cached


def _cache_set(fingerprint: str, store: Chroma) -> None:
    _VECTOR_STORE_CACHE[fingerprint] = store
    if fingerprint in _VECTOR_STORE_CACHE_ORDER:
        _VECTOR_STORE_CACHE_ORDER.remove(fingerprint)
    _VECTOR_STORE_CACHE_ORDER.append(fingerprint)
    while len(_VECTOR_STORE_CACHE_ORDER) > _VECTOR_STORE_CACHE_LIMIT:
        oldest = _VECTOR_STORE_CACHE_ORDER.pop(0)
        _VECTOR_STORE_CACHE.pop(oldest, None)


def _safe_collection_count(store: Chroma) -> int | None:
    try:
        collection = getattr(store, "_collection", None)
        if collection is None:
            return None
        return int(collection.count())
    except Exception:
        return None


def _open_persistent_store(collection_name: str, persist_dir: Path) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=_get_embeddings(),
    )


def _create_from_documents(
    documents: list[Document],
    *,
    collection_name: str,
    persist_dir: Path,
) -> Chroma:
    return Chroma.from_documents(
        documents=documents,
        embedding=_get_embeddings(),
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )


def create_vector_store(documents: list[Document]) -> Chroma:
    """Create/reuse a Chroma vector store with fingerprinted collections + cleanup."""
    if not documents:
        raise ValueError("documents cannot be empty.")

    fingerprint = _doc_fingerprint(documents)
    cached = _cache_get(fingerprint)
    if cached is not None:
        _VECTOR_STORE_STATS["cache_hits"] = int(_VECTOR_STORE_STATS.get("cache_hits", 0)) + 1
        _VECTOR_STORE_STATS["last_fingerprint"] = fingerprint
        return cached

    persist_dir = Path(CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    collection_name = _collection_name_from_fingerprint(fingerprint)
    prefix = (CHROMA_COLLECTION_PREFIX or "pdf_qa").strip() or "pdf_qa"
    _VECTOR_STORE_STATS["last_collection"] = collection_name
    _VECTOR_STORE_STATS["last_fingerprint"] = fingerprint

    try:
        store = _open_persistent_store(collection_name, persist_dir)
        count = _safe_collection_count(store)

        if count is None or count <= 0:
            store = _create_from_documents(
                documents,
                collection_name=collection_name,
                persist_dir=persist_dir,
            )
            _VECTOR_STORE_STATS["built_collections"] = int(_VECTOR_STORE_STATS.get("built_collections", 0)) + 1
        elif count != len(documents):
            client = getattr(store, "_client", None)
            if client is not None:
                _delete_collection(client, collection_name)
            store = _create_from_documents(
                documents,
                collection_name=collection_name,
                persist_dir=persist_dir,
            )
            _VECTOR_STORE_STATS["built_collections"] = int(_VECTOR_STORE_STATS.get("built_collections", 0)) + 1
        else:
            _VECTOR_STORE_STATS["reused_collections"] = int(_VECTOR_STORE_STATS.get("reused_collections", 0)) + 1

        client = getattr(store, "_client", None)
        if client is not None:
            _touch_collection_usage(persist_dir, collection_name)
            removed = _cleanup_old_collections(
                client,
                persist_dir,
                prefix=prefix,
                max_collections=CHROMA_MAX_COLLECTIONS,
                keep_collection_name=collection_name,
            )
            _VECTOR_STORE_STATS["cleanup_removed_collections"] = (
                int(_VECTOR_STORE_STATS.get("cleanup_removed_collections", 0)) + int(removed)
            )

        _cache_set(fingerprint, store)
        return store
    except Exception as exc:
        # Fallback for environments where sqlite file lock / I/O causes persistence failures.
        if "disk i/o error" in str(exc).lower():
            fallback_name = f"{collection_name}_{uuid4().hex[:8]}"
            store = Chroma.from_documents(
                documents=documents,
                embedding=_get_embeddings(),
                collection_name=fallback_name,
            )
            _VECTOR_STORE_STATS["fallback_builds"] = int(_VECTOR_STORE_STATS.get("fallback_builds", 0)) + 1
            _cache_set(fingerprint, store)
            return store
        raise


def get_vector_store_stats() -> dict[str, Any]:
    return {
        "cache_hits": int(_VECTOR_STORE_STATS.get("cache_hits", 0)),
        "reused_collections": int(_VECTOR_STORE_STATS.get("reused_collections", 0)),
        "built_collections": int(_VECTOR_STORE_STATS.get("built_collections", 0)),
        "cleanup_removed_collections": int(_VECTOR_STORE_STATS.get("cleanup_removed_collections", 0)),
        "fallback_builds": int(_VECTOR_STORE_STATS.get("fallback_builds", 0)),
        "last_collection": str(_VECTOR_STORE_STATS.get("last_collection", "") or ""),
        "last_fingerprint": str(_VECTOR_STORE_STATS.get("last_fingerprint", "") or ""),
        "cached_fingerprints": len(_VECTOR_STORE_CACHE),
    }


def get_retriever(vector_store: Chroma, top_k: int = 4) -> VectorStoreRetriever:
    """Return a retriever from an existing Chroma vector store."""
    return vector_store.as_retriever(search_kwargs={"k": top_k})
