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
OLLAMA_TEMPERATURE = _read_env_float("OLLAMA_TEMPERATURE", 0.2)
OLLAMA_NUM_PREDICT = _read_env_int("OLLAMA_NUM_PREDICT", 256)
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
TOP_K = _read_env_int("TOP_K", 4)
ENABLE_BROAD_SEARCH = _read_env_bool("ENABLE_BROAD_SEARCH", False)
BROAD_SEARCH_K = _read_env_int("BROAD_SEARCH_K", 8)

# Embedding
EMBEDDING_MODEL = _read_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB
CHROMA_PERSIST_DIR = _read_env("CHROMA_PERSIST_DIR", "./chroma_db")
