from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

from config import GEMINI_API_KEY, GOOGLE_ADK_MODEL
from rag.query_engine import (
    answer_with_document_index,
    extract_page_numbers,
    find_heading_matches,
    find_section_matches,
    hybrid_retrieve,
    is_page_query,
    summarize_page_docs,
)

try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    ADK_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback when dependency is absent
    Agent = None
    Runner = None
    InMemorySessionService = None
    types = None
    ADK_AVAILABLE = False


def _sanitize_proxy_for_google() -> None:
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    proxy_values = [os.getenv(key, "") for key in proxy_keys]
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


def _stringify_recent_conversation(conversation_state: list[tuple[str, str]] | None, limit: int = 3) -> str:
    pairs = conversation_state or []
    if not pairs:
        return "目前沒有先前對話。"

    lines: list[str] = []
    for question, answer in pairs[-limit:]:
        lines.append(f"使用者：{question}")
        lines.append(f"系統：{answer[:300]}")
    return "\n".join(lines)


def _preview_from_docs(docs: list[Any], limit: int = 4) -> str:
    if not docs:
        return "尚無檢索結果。"

    lines: list[str] = []
    for idx, doc in enumerate(docs[:limit], start=1):
        source = doc.metadata.get("source", "未知檔案")
        page = doc.metadata.get("page", "?")
        content = (doc.page_content or "").strip().replace("\n", " ")
        if len(content) > 180:
            content = content[:180] + "..."
        lines.append(f"{idx}. [{source} - 第 {page} 頁] {content}")
    return "\n\n".join(lines)


def _extract_text_from_event(event) -> str:
    if not getattr(event, "content", None) or not getattr(event.content, "parts", None):
        return ""
    parts: list[str] = []
    for part in event.content.parts:
        text = getattr(part, "text", "") or ""
        if text:
            parts.append(text)
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def run_adk_agent_answer(
    *,
    user_question: str,
    retriever,
    document_index: dict[str, Any] | None,
    all_docs: list[Any] | None,
    conversation_state: list[tuple[str, str]] | None = None,
    detected_mode: str | None = None,
) -> tuple[str, str]:
    """Run a Google ADK agent against the current PDF context."""
    if not ADK_AVAILABLE:
        raise RuntimeError("尚未安裝 google-adk，無法使用 Google ADK Agent。")
    if not GEMINI_API_KEY.strip():
        raise RuntimeError("尚未設定 GEMINI_API_KEY，Google ADK Agent 無法啟動。")

    _sanitize_proxy_for_google()
    os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)

    document_index = document_index or {}
    all_docs = all_docs or []
    tool_state: dict[str, Any] = {"preview": "尚無檢索結果。"}

    def search_document(query: str, top_k: int = 4) -> str:
        """搜尋與問題最相關的文件段落，回傳段落內容與頁碼。"""
        scored = hybrid_retrieve(query.strip(), all_docs, retriever, top_k=max(1, min(top_k, 6)))
        docs = [item.document if hasattr(item, "document") else item for item in scored]
        tool_state["preview"] = _preview_from_docs(docs)
        if not docs:
            return "找不到相關段落。"

        blocks: list[str] = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            text = (doc.page_content or "").strip()
            blocks.append(f"[第 {page} 頁]\n{text}")
        return "\n\n".join(blocks)

    def lookup_structured_answer(question: str) -> str:
        """用文件索引查找代碼、名稱、分類、清單或大綱型答案。"""
        answer = answer_with_document_index(question.strip(), document_index)
        if answer:
            heading_matches = find_heading_matches(question.strip(), document_index)[:4]
            pages = {item.get("page") for item in heading_matches if item.get("page")}
            if pages:
                docs = [
                    doc for doc in all_docs
                    if int(doc.metadata.get("page", 0) or 0) in {int(page) for page in pages}
                ]
                if docs:
                    tool_state["preview"] = _preview_from_docs(docs)
            return answer
        return "索引中找不到明確答案。"

    def get_page_content(page_number: int) -> str:
        """讀取指定頁面的完整內容，適合回答『第幾頁有什麼內容』。"""
        docs = [doc for doc in all_docs if int(doc.metadata.get("page", 0) or 0) == int(page_number)]
        tool_state["preview"] = _preview_from_docs(docs)
        if not docs:
            return f"文件中找不到第 {page_number} 頁。"
        return summarize_page_docs(docs)

    def get_document_outline() -> str:
        """整理文件大綱，回傳章節、頁碼與代表性條目。"""
        answer = answer_with_document_index("請整理這份文件的大綱。", document_index)
        if not answer:
            return "目前無法整理文件大綱。"
        section_matches = find_section_matches("大綱", document_index)[:6]
        pages = {int(item.get('page', 0) or 0) for item in section_matches if item.get("page")}
        docs = [doc for doc in all_docs if int(doc.metadata.get("page", 0) or 0) in pages]
        if docs:
            tool_state["preview"] = _preview_from_docs(docs)
        return answer

    recent_conversation = _stringify_recent_conversation(conversation_state)
    document_mode = detected_mode or "general"
    agent = Agent(
        name="pdf_qa_adk_agent",
        model=GOOGLE_ADK_MODEL,
        instruction=(
            "你是 PDF 問答系統中的 Google ADK Agent，請使用繁體中文回答。\n"
            "你只能根據工具回傳的文件內容與索引作答，不可以憑空猜測。\n"
            "回答前請先判斷題型並挑合適工具：\n"
            "1. 問頁碼內容時，先用 get_page_content。\n"
            "2. 問『是什麼』『代碼』『哪一類』『有哪些』『大綱』時，先用 lookup_structured_answer 或 get_document_outline。\n"
            "3. 若索引資訊不足，再用 search_document 補強。\n"
            "4. 回答時盡量附上頁碼；若工具已提供清單，請完整保留，不要漏項。\n"
            f"目前文件模式判定：{document_mode}。\n"
            f"最近對話摘要：\n{recent_conversation}"
        ),
        description="Answer questions about uploaded PDFs with tool-assisted reasoning.",
        tools=[search_document, lookup_structured_answer, get_page_content, get_document_outline],
    )

    app_name = "pdf_qa_adk"
    user_id = "gradio-user"
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(
        app_name=app_name,
        user_id=user_id,
        session_id=f"session-{uuid.uuid4().hex[:8]}",
    )
    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
    )

    if is_page_query(user_question):
        pages = extract_page_numbers(user_question)
        if pages:
            docs = [
                doc for doc in all_docs
                if int(doc.metadata.get("page", 0) or 0) in set(int(page) for page in pages)
            ]
            if docs:
                tool_state["preview"] = _preview_from_docs(docs)

    async def _collect_final_text() -> str:
        final = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.UserContent(parts=[types.Part.from_text(text=user_question)]),
        ):
            text = _extract_text_from_event(event)
            if text and event.is_final_response():
                final = text
        return final

    try:
        final_text = asyncio.run(_collect_final_text())
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "503" in message or "unavailable" in lowered or "high demand" in lowered:
            raise RuntimeError("Google ADK / Gemini 目前負載較高（503 UNAVAILABLE），請稍後再試。") from exc
        if "429" in message or "resource_exhausted" in lowered or "quota" in lowered:
            raise RuntimeError("Google ADK / Gemini 配額不足（429 RESOURCE_EXHAUSTED）。") from exc
        raise RuntimeError(f"Google ADK Agent 執行失敗：{message}") from exc

    if not final_text:
        raise RuntimeError("Google ADK Agent 沒有產生最終回答。")

    return final_text.strip(), tool_state["preview"]
