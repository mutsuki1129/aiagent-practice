from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_API_KEY, GEMINI_MODEL


def _sanitize_proxy_for_gemini() -> None:
    """Avoid known-invalid local proxy settings that block Gemini API calls."""
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    proxy_values = [os.getenv(k, "") for k in proxy_keys]
    bad_local_proxy = any("127.0.0.1:9" in value for value in proxy_values if value)
    if not bad_local_proxy:
        return

    for key in proxy_keys:
        os.environ.pop(key, None)

    no_proxy = os.getenv("NO_PROXY", "")
    required_hosts = ["generativelanguage.googleapis.com", "googleapis.com"]
    merged = [item.strip() for item in no_proxy.split(",") if item.strip()]
    for host in required_hosts:
        if host not in merged:
            merged.append(host)
    os.environ["NO_PROXY"] = ",".join(merged)


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat model client."""
    if not GEMINI_API_KEY.strip():
        raise RuntimeError(
            "尚未設定 GEMINI_API_KEY。請在 .env 填入你的 Google AI Studio API Key。"
        )

    _sanitize_proxy_for_gemini()

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
    )
