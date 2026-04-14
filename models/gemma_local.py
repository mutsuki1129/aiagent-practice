from __future__ import annotations

import threading
import time

import requests
from langchain_ollama import ChatOllama

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_HEALTHCHECK_TTL_SECONDS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_NUM_THREAD,
    OLLAMA_TEMPERATURE,
)

_OLLAMA_LOCK = threading.Lock()
_OLLAMA_LLM_INSTANCE: ChatOllama | None = None
_LAST_HEALTHCHECK_TS = 0.0


def _check_ollama_health(timeout: float = 3.0, force: bool = False) -> None:
    """Validate Ollama service and model availability before creating the LLM client."""
    global _LAST_HEALTHCHECK_TS

    now = time.time()
    if not force and _LAST_HEALTHCHECK_TS and (now - _LAST_HEALTHCHECK_TS) < OLLAMA_HEALTHCHECK_TTL_SECONDS:
        return

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

    _LAST_HEALTHCHECK_TS = now


def get_gemma_llm(num_predict_override: int | None = None) -> ChatOllama:
    """Return a configured Gemma ChatOllama instance."""
    global _OLLAMA_LLM_INSTANCE

    if num_predict_override is not None:
        _check_ollama_health()
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=OLLAMA_TEMPERATURE,
            num_predict=num_predict_override,
            num_ctx=OLLAMA_NUM_CTX,
            num_thread=OLLAMA_NUM_THREAD,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )

    if _OLLAMA_LLM_INSTANCE is not None:
        _check_ollama_health()
        return _OLLAMA_LLM_INSTANCE

    with _OLLAMA_LOCK:
        if _OLLAMA_LLM_INSTANCE is None:
            _check_ollama_health(force=True)
            _OLLAMA_LLM_INSTANCE = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=OLLAMA_TEMPERATURE,
                num_predict=OLLAMA_NUM_PREDICT,
                num_ctx=OLLAMA_NUM_CTX,
                num_thread=OLLAMA_NUM_THREAD,
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
        else:
            _check_ollama_health()

    return _OLLAMA_LLM_INSTANCE
