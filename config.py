import os
from dotenv import load_dotenv

load_dotenv()


def _read_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    # Handle UTF-8 BOM accidentally saved in .env key names.
    return os.getenv("\ufeff" + name, default)


def _read_env_int(name: str, default: int) -> int:
    raw = _read_env(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _read_env_float(name: str, default: float) -> float:
    raw = _read_env(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default


def _read_env_bool(name: str, default: bool) -> bool:
    raw = _read_env(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


# Ollama
OLLAMA_BASE_URL = _read_env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _read_env("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_GEMMA_MODEL = _read_env("OLLAMA_GEMMA_MODEL", OLLAMA_MODEL or "gemma4:e4b")
OLLAMA_QWEN_MODEL = _read_env("OLLAMA_QWEN_MODEL", "qwen3:14b")
OLLAMA_TEMPERATURE = _read_env_float("OLLAMA_TEMPERATURE", 0.2)
OLLAMA_NUM_PREDICT = _read_env_int("OLLAMA_NUM_PREDICT", 512)
OLLAMA_NUM_CTX = _read_env_int("OLLAMA_NUM_CTX", 4096)
OLLAMA_NUM_THREAD = _read_env_int(
    "OLLAMA_NUM_THREAD",
    max((os.cpu_count() or 4) - 1, 1),
)
OLLAMA_KEEP_ALIVE = _read_env("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_HEALTHCHECK_TTL_SECONDS = _read_env_int("OLLAMA_HEALTHCHECK_TTL_SECONDS", 30)

# Gemini
GEMINI_API_KEY = _read_env("GEMINI_API_KEY", "")
GEMINI_MODEL = _read_env("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_ADK_MODEL = _read_env("GOOGLE_ADK_MODEL", GEMINI_MODEL or "gemini-2.5-flash")

# RAG
CHUNK_SIZE = _read_env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _read_env_int("CHUNK_OVERLAP", 200)
TOP_K = _read_env_int("TOP_K", 6)
ENABLE_BROAD_SEARCH = _read_env_bool("ENABLE_BROAD_SEARCH", False)
BROAD_SEARCH_K = _read_env_int("BROAD_SEARCH_K", 8)
RRF_RANK_CONSTANT = _read_env_int("RRF_RANK_CONSTANT", 60)
QUERY_VARIANT_COUNT = _read_env_int("QUERY_VARIANT_COUNT", 8)
ENABLE_RETRIEVAL_CACHE = _read_env_bool("ENABLE_RETRIEVAL_CACHE", True)
RETRIEVAL_CACHE_SIZE = _read_env_int("RETRIEVAL_CACHE_SIZE", 128)
RETRIEVAL_PROFILE = _read_env("RETRIEVAL_PROFILE", "auto").strip().lower()
COMPLEX_REASONING_PASSES = _read_env_int("COMPLEX_REASONING_PASSES", 2)
COMPLEX_REASONING_MIN_QUESTION_LEN = _read_env_int("COMPLEX_REASONING_MIN_QUESTION_LEN", 30)
COMPLEX_REASONING_MIN_CONTEXT_DOCS = _read_env_int("COMPLEX_REASONING_MIN_CONTEXT_DOCS", 7)
COMPLEX_RETRIEVAL_TOP_K_BOOST = _read_env_int("COMPLEX_RETRIEVAL_TOP_K_BOOST", 2)
ENABLE_BM25_RETRIEVAL = _read_env_bool("ENABLE_BM25_RETRIEVAL", True)
BM25_CANDIDATE_K = _read_env_int("BM25_CANDIDATE_K", 24)
ENABLE_CROSS_ENCODER_RERANK = _read_env_bool("ENABLE_CROSS_ENCODER_RERANK", False)
CROSS_ENCODER_MODEL = _read_env("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_TOP_K = _read_env_int("CROSS_ENCODER_TOP_K", 12)

# Embedding
EMBEDDING_MODEL = _read_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB
CHROMA_PERSIST_DIR = _read_env("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_PREFIX = _read_env("CHROMA_COLLECTION_PREFIX", "pdf_qa")
CHROMA_MAX_COLLECTIONS = _read_env_int("CHROMA_MAX_COLLECTIONS", 24)
