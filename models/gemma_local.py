from __future__ import annotations

import threading
import time

import requests
from langchain_ollama import ChatOllama

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_GEMMA_MODEL,
    OLLAMA_HEALTHCHECK_TTL_SECONDS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_NUM_THREAD,
    OLLAMA_TEMPERATURE,
)

_OLLAMA_LOCK = threading.Lock()
_OLLAMA_LLM_INSTANCES: dict[str, ChatOllama] = {}
_LAST_HEALTHCHECK_TS = 0.0
_LAST_MODEL_SET: set[str] = set()


def _check_ollama_health(required_model: str, timeout: float = 3.0, force: bool = False) -> None:
    """Validate Ollama service and model availability before creating the LLM client."""
    global _LAST_HEALTHCHECK_TS
    global _LAST_MODEL_SET

    now = time.time()
    if (
        not force
        and _LAST_HEALTHCHECK_TS
        and (now - _LAST_HEALTHCHECK_TS) < OLLAMA_HEALTHCHECK_TTL_SECONDS
        and required_model in _LAST_MODEL_SET
    ):
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
    if required_model not in model_names:
        raise RuntimeError(
            f"Ollama 已啟動，但找不到模型 `{required_model}`。"
            f"請先執行：ollama pull {required_model}"
        )

    _LAST_HEALTHCHECK_TS = now
    _LAST_MODEL_SET = model_names


def get_ollama_llm(model_name: str | None = None, num_predict_override: int | None = None) -> ChatOllama:
    """Return a configured local ChatOllama instance for the requested model."""
    target_model = (model_name or OLLAMA_GEMMA_MODEL).strip() or OLLAMA_GEMMA_MODEL

    if num_predict_override is not None:
        _check_ollama_health(required_model=target_model)
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=target_model,
            temperature=OLLAMA_TEMPERATURE,
            num_predict=num_predict_override,
            num_ctx=OLLAMA_NUM_CTX,
            num_thread=OLLAMA_NUM_THREAD,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )

    cached = _OLLAMA_LLM_INSTANCES.get(target_model)
    if cached is not None:
        _check_ollama_health(required_model=target_model)
        return cached

    with _OLLAMA_LOCK:
        cached = _OLLAMA_LLM_INSTANCES.get(target_model)
        if cached is None:
            _check_ollama_health(required_model=target_model, force=True)
            cached = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=target_model,
                temperature=OLLAMA_TEMPERATURE,
                num_predict=OLLAMA_NUM_PREDICT,
                num_ctx=OLLAMA_NUM_CTX,
                num_thread=OLLAMA_NUM_THREAD,
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
            _OLLAMA_LLM_INSTANCES[target_model] = cached
        else:
            _check_ollama_health(required_model=target_model)

    return cached


def get_gemma_llm(num_predict_override: int | None = None) -> ChatOllama:
    """Backward-compatible helper for Gemma local model."""
    return get_ollama_llm(model_name=OLLAMA_GEMMA_MODEL, num_predict_override=num_predict_override)
