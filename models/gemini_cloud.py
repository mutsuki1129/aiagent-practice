from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_API_KEY, GEMINI_MODEL


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat model client."""
    if not GEMINI_API_KEY.strip():
        raise RuntimeError(
            "尚未設定 GEMINI_API_KEY。請在 .env 填入你的 Google AI Studio API Key。"
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
    )
