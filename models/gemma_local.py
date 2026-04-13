from __future__ import annotations

import requests
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


def _check_ollama_health(timeout: float = 3.0) -> None:
    """Validate Ollama service and model availability before creating the LLM client."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "無法連線到 Ollama 服務，請先確認 Ollama 已啟動（可執行 `ollama serve`），"
            f"目前嘗試連線位址：{OLLAMA_BASE_URL}"
        ) from exc

    model_names = {
        model.get("name", "")
        for model in response.json().get("models", [])
        if isinstance(model, dict)
    }
    if OLLAMA_MODEL not in model_names:
        raise RuntimeError(
            f"Ollama 已啟動，但找不到模型 `{OLLAMA_MODEL}`。"
            f"請先執行：ollama pull {OLLAMA_MODEL}"
        )


def get_gemma_llm() -> ChatOllama:
    """Return a configured Gemma ChatOllama instance."""
    _check_ollama_health()
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.3,
    )
