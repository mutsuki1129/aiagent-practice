import os
from dotenv import load_dotenv

load_dotenv()


def _read_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    # Handle UTF-8 BOM accidentally saved in .env key names.
    return os.getenv("\ufeff" + name, default)


# Ollama 設定
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"

# Gemini 設定
GEMINI_API_KEY = _read_env("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# RAG 設定
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# Embedding 設定
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB 設定
CHROMA_PERSIST_DIR = "./chroma_db"
