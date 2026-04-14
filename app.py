from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter
import re
import time

import gradio as gr
import requests
from langchain_core.documents import Document

from agents.google_adk_agent import ADK_AVAILABLE, run_adk_agent_answer
from config import (
    BROAD_SEARCH_K,
    ENABLE_BROAD_SEARCH,
    GEMINI_API_KEY,
    GOOGLE_ADK_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT,
    TOP_K,
)
from models.gemma_local import get_gemma_llm
from models.gemini_cloud import get_gemini_llm
from rag.pdf_loader import load_and_split
from rag.query_engine import (
    answer_with_document_index,
    build_document_index,
    build_llm_prompt,
    build_summary_prompt,
    detect_document_mode,
    extract_lookup_terms,
    extract_page_numbers,
    find_heading_matches,
    find_section_matches,
    hybrid_retrieve,
    is_code_or_lookup_query,
    is_page_query,
    is_summary_query,
    normalize_text,
    should_use_literal_lookup,
    summarize_page_docs,
)
from rag.vector_store import create_vector_store, get_retriever

_MODEL_STATUS_CACHE: str | None = None
_MODEL_STATUS_LAST_TS = 0.0
_MODEL_STATUS_TTL_SECONDS = 5.0
_EXPANDABLE_THRESHOLD = 350


def _friendly_model_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "google adk agent" in lowered or "google_adk" in lowered:
        return message
    if "429" in message or "resource_exhausted" in lowered or "quota" in lowered:
        return "Gemini 配額不足（429 RESOURCE_EXHAUSTED）。請稍後再試，或先改用 Gemma 本地模型。"
    if "winerror 10061" in lowered or "connecterror" in lowered or "failed to establish a new connection" in lowered:
        return "Gemini 連線失敗（可能是代理設定或網路限制）。已建議改用 Gemma 本地模型。"
    if "api key" in lowered or "permission" in lowered or "unauthorized" in lowered:
        return "Gemini 驗證失敗，請確認 API Key 是否正確且可用。"
    return f"模型呼叫失敗：{message}"


def _resolve_llm(model_choice: str, num_predict_override: int | None = None):
    if model_choice.startswith("Gemma"):
        return get_gemma_llm(num_predict_override=num_predict_override)
    return get_gemini_llm()


def check_model_status() -> str:
    """Return markdown status for Ollama and Gemini availability."""
    global _MODEL_STATUS_CACHE
    global _MODEL_STATUS_LAST_TS

    now = time.time()
    if _MODEL_STATUS_CACHE is not None and (now - _MODEL_STATUS_LAST_TS) < _MODEL_STATUS_TTL_SECONDS:
        return _MODEL_STATUS_CACHE

    ollama_status = "不可用"
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.ok:
            models = {m.get("name", "") for m in response.json().get("models", []) if isinstance(m, dict)}
            if OLLAMA_MODEL in models:
                ollama_status = f"已連線（{OLLAMA_MODEL} 可用）"
            else:
                ollama_status = f"已連線（但缺少模型 {OLLAMA_MODEL}）"
        else:
            ollama_status = "服務異常"
    except requests.RequestException:
        ollama_status = "未連線（請確認 ollama serve）"

    gemini_status = "API Key 已設定（未即時檢測配額/連線）" if GEMINI_API_KEY.strip() else "未設定 GEMINI_API_KEY"
    adk_status = f"已安裝（模型：{GOOGLE_ADK_MODEL}）" if ADK_AVAILABLE else "未安裝 google-adk"

    status = (
        "### 模型狀態\n"
        f"- Ollama / Gemma 4：{ollama_status}\n"
        f"- Gemini：{gemini_status}\n"
        f"- Google ADK Agent：{adk_status}"
    )
    _MODEL_STATUS_CACHE = status
    _MODEL_STATUS_LAST_TS = now
    return status


def _format_retrieved_preview(source_docs: list[Any]) -> str:
    if not source_docs:
        return "尚無檢索結果。"

    lines: list[str] = []
    for idx, doc in enumerate(source_docs[:4], start=1):
        source = doc.metadata.get("source", "未知檔案")
        page = doc.metadata.get("page", "?")
        content = (doc.page_content or "").strip().replace("\n", " ")
        if len(content) > 220:
            content = content[:220] + "..."
        lines.append(f"{idx}. [{source} - 第 {page} 頁] {content}")

    return "\n\n".join(lines)


def _normalize_text(text: str) -> str:
    """Normalize text for robust exact-substring matching."""
    return normalize_text(text)


def _find_exact_matches(query: str, docs: list[Any]) -> list[Any]:
    normalized_query = _normalize_text(query)
    if len(normalized_query) < 2:
        return []

    matches: list[Any] = []
    for doc in docs:
        content = _normalize_text(str(getattr(doc, "page_content", "") or ""))
        if normalized_query in content:
            matches.append(doc)
    return matches


def _is_summary_like_question(user_question: str) -> bool:
    normalized = _normalize_text(user_question)
    summary_markers = (
        "大意",
        "摘要",
        "總結",
        "總覽",
        "概述",
        "概觀",
        "大綱",
        "重點",
        "主要在講什麼",
        "主要說什麼",
        "在說什麼",
        "內容是什麼",
        "what is this document about",
        "summarize",
        "summary",
        "outline",
        "overview",
    )
    return is_summary_query(user_question) or any(marker in normalized for marker in summary_markers)


def _is_general_inference_question(user_question: str) -> bool:
    normalized = _normalize_text(user_question)
    markers = (
        "身分",
        "身份",
        "角色",
        "職位",
        "工作",
        "職能",
        "談話者",
        "說話者",
        "人物關係",
        "關係",
        "會議重點",
        "重點整理",
        "會議",
        "討論重點",
        "who are the speakers",
        "identity of the speakers",
        "job functions",
        "roles",
        "working on",
        "likely doing",
        "relationship",
        "meeting highlights",
        "key discussion points",
    )
    return any(marker in normalized for marker in markers)


def _invoke_llm_text(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = str(getattr(response, "content", "") or "").strip()
    if content:
        return content
    return ""


def _needs_candidate_rewrite(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    failure_markers = (
        "根據文件內容，我找不到相關資訊",
        "我找不到相關資訊",
        "沒有提供",
        "並沒有提供",
        "無法確認",
    )
    return any(marker in text for marker in failure_markers)


def _finalize_hybrid_answer(answer: str, candidate_hint: str) -> str:
    text = (answer or "").strip()
    hint = (candidate_hint or "").strip()
    if hint and _needs_candidate_rewrite(text):
        return hint
    return text


def _to_expandable_answer(answer: str, threshold: int = _EXPANDABLE_THRESHOLD) -> str:
    text = (answer or "").strip()
    if len(text) <= threshold:
        return text

    one_line = " ".join(text.split())
    summary = one_line[:120] + ("..." if len(one_line) > 120 else "")
    return (
        f"<details><summary>{summary}</summary>\n\n"
        f"{text}\n\n"
        f"</details>"
    )


def _build_exact_match_answer(query: str, docs: list[Any]) -> str:
    snippets: list[str] = []
    pages: set[str] = set()
    for doc in docs[:3]:
        page = doc.metadata.get("page", "?")
        pages.add(f"第 {page} 頁")
        content = (doc.page_content or "").strip().replace("\n", " ")
        if len(content) > 180:
            content = content[:180] + "..."
        snippets.append(f"- {content}")

    lines = [
        "已在文件中找到相符內容。",
        f"查詢文字：{query.strip()}",
    ]
    if snippets:
        lines.append("\n相符段落：")
        lines.extend(snippets)
    if pages:
        lines.append(f"\n參考來源： {', '.join(sorted(pages))}")
    return "\n".join(lines)


def _dedupe_docs(docs: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[tuple[Any, Any, str]] = set()
    for doc in docs:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            (doc.page_content or "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _excerpt_page_content(text: str, user_question: str, max_chars: int = 420) -> str:
    content = (text or "").strip()
    if len(content) <= max_chars:
        return content

    lookup_terms = [term for term in extract_lookup_terms(user_question) if len(term.strip()) >= 2]
    normalized_content = _normalize_text(content)
    for term in lookup_terms:
        normalized_term = _normalize_text(term)
        if not normalized_term:
            continue
        hit = normalized_content.find(normalized_term)
        if hit >= 0:
            start = max(0, hit - 120)
            end = min(len(content), start + max_chars)
            excerpt = content[start:end].strip()
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(content):
                excerpt = excerpt + "..."
            return excerpt

    return content[:max_chars].strip() + "..."


def _compress_context_docs(user_question: str, docs: list[Any], *, max_docs: int, max_chars: int) -> list[Document]:
    compressed: list[Document] = []
    for doc in docs[:max_docs]:
        compressed.append(
            Document(
                page_content=_excerpt_page_content(doc.page_content or "", user_question, max_chars=max_chars),
                metadata=dict(getattr(doc, "metadata", {}) or {}),
            )
        )
    return compressed


def _select_reasoning_docs(
    user_question: str,
    all_docs: list[Any],
    retriever,
    document_index: dict[str, Any],
    *,
    top_k: int,
) -> list[Any]:
    selected: list[Any] = []

    heading_matches = find_heading_matches(user_question, document_index)[:8]
    matched_pages = {int(item.get("page", 0) or 0) for item in heading_matches if item.get("page")}
    if matched_pages:
        selected.extend(
            doc for doc in all_docs if int(doc.metadata.get("page", 0) or 0) in matched_pages
        )

    section_matches = find_section_matches(user_question, document_index)[:3]
    section_titles = {item.get("title", "") for item in section_matches if item.get("title")}
    if section_titles:
        section_pages = {
            int(item.get("page", 0) or 0)
            for item in document_index.get("headings", [])
            if item.get("section", "") in section_titles
        }
        selected.extend(
            doc for doc in all_docs if int(doc.metadata.get("page", 0) or 0) in section_pages
        )

    lookup_terms = extract_lookup_terms(user_question)
    for term in lookup_terms[:4]:
        normalized_term = _normalize_text(term)
        if not normalized_term or len(normalized_term) < 2:
            continue
        selected.extend(
            doc
            for doc in all_docs
            if normalized_term in _normalize_text(str(getattr(doc, "page_content", "") or ""))
        )

    scored_docs = hybrid_retrieve(user_question.strip(), all_docs, retriever, top_k=max(top_k, 6))
    selected.extend(item.document if hasattr(item, "document") else item for item in scored_docs)

    return _dedupe_docs(selected)[:top_k]


def _build_reasoning_prompt(
    user_question: str,
    context_docs: list[Any],
    document_index: dict[str, Any],
    detected_mode: str | None,
    candidate_hint: str = "",
) -> str:
    question = user_question.strip()
    heading_matches = find_heading_matches(question, document_index)[:8]
    section_matches = find_section_matches(question, document_index)[:4]
    candidate_lines: list[str] = []
    for item in heading_matches:
        candidate_lines.append(
            f"- 候選條目：{item.get('code', '?')} {item.get('title', '')} / 分類：{item.get('section', '未知')} / 第 {item.get('page', '?')} 頁"
        )
    for item in section_matches:
        candidate_lines.append(
            f"- 候選分類：{item.get('title', '')} / 起始代碼：{item.get('start_code', '?')} / 第 {item.get('page', '?')} 頁"
        )
    candidate_block = "\n".join(candidate_lines) if candidate_lines else "- 無明確候選條目，請只根據段落內容判斷。"

    context_blocks: list[str] = []
    for doc in context_docs[:10]:
        page = doc.metadata.get("page", "?")
        text = (doc.page_content or "").strip()
        context_blocks.append(f"[第 {page} 頁]\n{text}")
    joined_context = "\n\n".join(context_blocks)

    hint_block = candidate_hint.strip() or candidate_block

    if any(marker in question for marker in ("有哪些", "有哪一些", "相關分類")):
        return (
            "你是 PDF 文件問答助手，請只根據提供段落回答，使用繁體中文。\n"
            "這是一題『完整列舉』問題。你必須列出段落中所有符合條件的條目，不可以只回其中一筆。\n"
            "若段落中出現多個代碼與名稱，請全部整理成清單，每行包含代碼、名稱、頁碼與所屬分類。\n"
            "你可以參考候選答案草稿，但最終答案必須由你根據文件段落整理後輸出。\n"
            "若證據不足，再明確說找不到，不要憑空猜測。\n\n"
            f"候選答案草稿：\n{hint_block}\n\n"
            f"候選資訊：\n{candidate_block}\n\n"
            f"文件段落：\n{joined_context}\n\n"
            f"問題：{question}"
        )
    if any(marker in question for marker in ("是什麼", "是甚麼", "代碼", "哪一類")):
        return (
            "你是 PDF 文件問答助手，請只根據提供段落回答，使用繁體中文。\n"
            "這是一題『單一條目說明』問題。請優先回答名稱、代碼、所屬分類、頁碼與定義。\n"
            "你可以參考候選答案草稿，但最終答案必須由你根據文件段落整理後輸出。\n"
            "如果段落中其實已經出現對應條目，就不要回答找不到。\n"
            "若其中某欄位無法確認，才明確說該欄位缺少依據。\n\n"
            f"候選答案草稿：\n{hint_block}\n\n"
            f"候選資訊：\n{candidate_block}\n\n"
            f"文件段落：\n{joined_context}\n\n"
            f"問題：{question}"
        )
    if _is_index_document_mode(detected_mode):
        return (
            build_llm_prompt(user_question, context_docs, document_index)
            + f"\n\n候選答案草稿：\n{hint_block}\n\n候選資訊：\n{candidate_block}"
        )

    return (
        "你是 PDF 文件問答助手，請使用繁體中文，只根據提供的文件內容回答。\n"
        "請優先整理與問題最相關的段落，必要時附上頁碼。"
        "不要把這份文件誤判成 ICD 或其他特定分類手冊，除非段落中真的有相關內容。\n"
        "若證據不足，請坦白說明，不要套用固定模板。\n\n"
        f"文件段落：\n{joined_context}\n\n"
        f"問題：{question}"
    )


def _prepare_reasoning_context(
    user_question: str,
    context_docs: list[Any],
    *,
    is_list_query: bool,
    is_lookup_query: bool,
    is_page_lookup: bool,
) -> list[Document]:
    if is_page_lookup:
        return _compress_context_docs(user_question, context_docs, max_docs=2, max_chars=700)
    if is_list_query:
        return _compress_context_docs(user_question, context_docs, max_docs=6, max_chars=280)
    if is_lookup_query:
        return _compress_context_docs(user_question, context_docs, max_docs=4, max_chars=340)
    return _compress_context_docs(user_question, context_docs, max_docs=5, max_chars=320)


def _is_index_document_mode(detected_mode: str | None) -> bool:
    return (detected_mode or "").strip().lower() == "index"


def _build_generic_summary_prompt(user_question: str, all_docs: list[Any]) -> str:
    intro_blocks: list[str] = []
    for doc in all_docs[: min(len(all_docs), 6)]:
        page = doc.metadata.get("page", "?")
        text = (doc.page_content or "").strip().replace("\n", " ")
        if len(text) > 360:
            text = text[:360] + "..."
        intro_blocks.append(f"[第 {page} 頁] {text}")

    context = "\n\n".join(intro_blocks) if intro_blocks else "目前沒有可用的文件內容。"
    return (
        "你是 PDF 文件摘要助手，請使用繁體中文，只根據提供的文件內容回答。\n"
        "如果使用者在問這份文件的大意、摘要、總結、概述或在說什麼，"
        "請整理主題與重點，不要把文件硬套成 ICD 或其他特定領域。\n"
        "若文件內容不足以完整概括，就明確說明目前能確認的內容。\n\n"
        f"文件片段：\n{context}\n\n"
        f"問題：{user_question.strip()}"
    )


def _build_general_summary_fallback(all_docs: list[Any]) -> str:
    page_count = len({int(doc.metadata.get("page", 0) or 0) for doc in all_docs if doc.metadata.get("page")})
    sample_docs = all_docs[: min(len(all_docs), 6)]
    text = "\n".join((doc.page_content or "").strip() for doc in sample_docs)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    speaker_counts: Counter[str] = Counter()
    for line in lines:
        match = re.match(r"^([A-Za-z][A-Za-z0-9 _-]{0,24}|[\u4e00-\u9fff]{1,8})\s*:\s+", line)
        if match:
            speaker = match.group(1).strip()
            if len(speaker) >= 2:
                speaker_counts[speaker] += 1

    english_stopwords = {
        "that", "this", "with", "from", "have", "will", "your", "about", "after",
        "before", "there", "their", "would", "could", "should", "today", "really",
        "still", "great", "thanks", "hello", "line", "speaker", "reply", "understand",
        "point", "another", "complete", "sentence", "clear", "statement", "number",
    }
    word_counts: Counter[str] = Counter()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]{3,}", text.lower()):
        if token in english_stopwords:
            continue
        word_counts[token] += 1

    top_keywords = [word for word, _ in word_counts.most_common(6)]
    top_speakers = [name for name, _ in speaker_counts.most_common(4)]

    lines_out = [f"這份文件共 {page_count or len(sample_docs)} 頁，主要內容如下："]
    if top_speakers:
        lines_out.append(f"- 內容型態：以對話或逐句交流為主，主要說話者包含 {', '.join(top_speakers)}。")
    else:
        lines_out.append("- 內容型態：一般文件或逐段文字內容。")

    if top_keywords:
        lines_out.append(f"- 核心主題：可從前幾頁看出重點圍繞 {', '.join(top_keywords)}。")

    if lines:
        preview_line = lines[0]
        if len(preview_line) > 180:
            preview_line = preview_line[:180] + "..."
        lines_out.append(f"- 文件開頭重點：{preview_line}")

    lines_out.append("- 若你要更精準，我也可以再進一步整理成章節摘要、人物重點或會議重點。")
    return "\n".join(lines_out)


def _build_general_inference_fallback(user_question: str, context_docs: list[Any]) -> str:
    text = "\n".join((doc.page_content or "").strip() for doc in context_docs[:6])
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    speaker_counts: Counter[str] = Counter()
    for line in lines:
        match = re.match(r"^([A-Za-z][A-Za-z0-9 _-]{0,24}|[\u4e00-\u9fff]{1,8})\s*:\s+", line)
        if match:
            speaker = match.group(1).strip()
            if len(speaker) >= 2:
                speaker_counts[speaker] += 1

    lowered = text.lower()
    software_markers = (
        "deploy", "bug", "pull request", "cache", "regression", "feature flag",
        "rollback", "production", "dashboard", "prompt", "model", "qa",
        "tests", "release", "architecture", "sync", "roadmap", "ticket",
    )
    meeting_markers = ("meeting", "sync", "agenda", "slides", "invite", "q and a")
    domain_hits = [marker for marker in software_markers if marker in lowered]
    meeting_hits = [marker for marker in meeting_markers if marker in lowered]

    top_speakers = [name for name, _ in speaker_counts.most_common(4)]
    speaker_text = ", ".join(top_speakers) if top_speakers else "文件中的說話者"

    question_normalized = _normalize_text(user_question)
    lines_out: list[str] = []

    if "meeting highlights" in question_normalized or "會議重點" in question_normalized or "討論重點" in question_normalized or "重點整理" in question_normalized:
        lines_out.append(f"從對話判斷，{speaker_text} 討論的重點偏向工作協作與專案執行。")
        if domain_hits:
            lines_out.append(f"- 工作主題：涉及 {', '.join(domain_hits[:6])}。")
        if meeting_hits:
            lines_out.append(f"- 協作情境：提到 {', '.join(meeting_hits[:3])}，顯示他們在準備會議、同步進度或安排工作。")
        if len(top_speakers) >= 2:
            lines_out.append(f"- 互動模式：{top_speakers[0]} 與 {top_speakers[1]} 的對話像是同一團隊成員之間的工作交流。")
    elif "relationship" in question_normalized or "關係" in question_normalized:
        if len(top_speakers) >= 2:
            lines_out.append(f"從對話內容看，{top_speakers[0]} 與 {top_speakers[1]} 比較像同事、專案夥伴，或同一團隊成員。")
        else:
            lines_out.append("從對話內容看，說話者之間像是在合作處理共同任務的關係。")
        if domain_hits or meeting_hits:
            lines_out.append("- 依據：他們反覆討論部署、測試、回滾、會議與工作分工，互動模式較像工作夥伴而非單純朋友聊天。")
    elif "working on" in question_normalized or "工作" in question_normalized:
        if domain_hits:
            lines_out.append(f"從對話內容判斷，{speaker_text} 很可能正在處理軟體產品或 AI/PDF 問答系統相關工作。")
            lines_out.append(f"- 依據：文中提到 {', '.join(domain_hits[:6])}。")
        else:
            lines_out.append(f"從目前段落看，{speaker_text} 正在討論某個共同任務或專案，但領域證據還不夠明確。")
    elif _is_general_inference_question(user_question):
        if domain_hits:
            lines_out.append(f"從對話判斷，{speaker_text} 很可能是軟體或 AI 產品團隊成員。")
            if len(top_speakers) >= 2:
                lines_out.append(f"- 推測：{top_speakers[0]} 與 {top_speakers[1]} 可能是工程師、QA、技術 PM，或負責產品交付的人員。")
            lines_out.append(f"- 依據：他們談到 {', '.join(domain_hits[:6])}。")
            if meeting_hits:
                lines_out.append(f"- 另外也提到 {', '.join(meeting_hits[:3])}，顯示他們在協作與規劃工作。")
        else:
            lines_out.append(f"從目前內容只能看出 {speaker_text} 是在進行日常對話或工作交流，無法非常精準判定正式身分。")
    else:
        return ""

    if lines:
        evidence = lines[0]
        if len(evidence) > 180:
            evidence = evidence[:180] + "..."
        lines_out.append(f"- 片段依據：{evidence}")

    return "\n".join(lines_out)


def _should_use_candidate_directly(
    candidate_hint: str,
    *,
    detected_mode: str | None,
    is_summary: bool,
    is_list_query: bool,
    is_lookup_query: bool,
    is_page_lookup: bool,
) -> bool:
    if not candidate_hint.strip():
        return False
    if not _is_index_document_mode(detected_mode):
        return False
    if is_page_lookup:
        return True
    if is_summary:
        return True
    return is_list_query or is_lookup_query


def _gemma_num_predict_for_query(
    *,
    is_summary: bool,
    is_list_query: bool,
    is_lookup_query: bool,
    is_page_lookup: bool,
) -> int | None:
    if is_summary:
        return max(384, min(OLLAMA_NUM_PREDICT, 512))
    if is_list_query:
        return min(OLLAMA_NUM_PREDICT, 224)
    if is_lookup_query or is_page_lookup:
        return min(OLLAMA_NUM_PREDICT, 192)
    return min(OLLAMA_NUM_PREDICT, 256)


def _should_bypass_llm_for_query(user_question: str) -> bool:
    question = user_question.strip()
    return (
        is_page_query(question)
        or is_code_or_lookup_query(question)
        or should_use_literal_lookup(question)
    )


def _resolve_query_mode(mode_choice: str, detected_mode: str) -> str:
    if mode_choice == "通用推理模式":
        return "general"
    return "general"


def _format_doc_mode_status(detected_mode: str) -> str:
    if detected_mode == "index":
        return "### 文件模式\n- 自動判定：`索引型 PDF`\n- 目前策略：`混合式`\n- 說明：先讓模型根據檢索內容生成；若模型失手，再退回結構化候選答案。"
    return "### 文件模式\n- 自動判定：`一般 PDF`\n- 目前策略：`混合式`\n- 說明：保留 RAG + LLM 推理，必要時以結構化候選答案補強。"



def process_uploaded_pdfs(files: list[str] | None):
    if not files:
        return (
            None,
            {},
            "請先上傳至少一份 PDF。",
            check_model_status(),
            _format_doc_mode_status("general"),
            "尚無檢索結果。",
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
            [],
            "general",
        )

    all_docs = []
    document_index: dict[str, Any] = {}
    unique_pages: set[tuple[str, int]] = set()
    used_files: list[str] = []

    try:
        for file_path in files:
            docs = load_and_split(file_path)
            all_docs.extend(docs)
            used_files.append(Path(file_path).name)
            for doc in docs:
                source = str(doc.metadata.get("source", ""))
                page = int(doc.metadata.get("page", 0))
                if source and page > 0:
                    unique_pages.add((source, page))

        document_index = build_document_index(all_docs)
        detected_mode = detect_document_mode(document_index, all_docs)
        vector_store = create_vector_store(all_docs)
        retriever = get_retriever(vector_store, top_k=TOP_K)

        status = (
            f"已載入 {len(unique_pages)} 頁，切割為 {len(all_docs)} 個段落。"
            f"\n檔案：{', '.join(used_files)}"
        )
        return (
            retriever,
            document_index,
            status,
            check_model_status(),
            _format_doc_mode_status(detected_mode),
            "尚無檢索結果。",
            gr.update(interactive=True),
            gr.update(interactive=True),
            all_docs,
            [],
            detected_mode,
        )
    except Exception as exc:
        return (
            None,
            {},
            f"PDF 處理失敗：{exc}",
            check_model_status(),
            _format_doc_mode_status("general"),
            "尚無檢索結果。",
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
            [],
            "general",
        )


def ask_question(
    user_question: str,
    chat_history: list[dict[str, str]] | None,
    retriever,
    document_index_state: dict[str, Any] | None,
    model_choice: str,
    mode_choice: str,
    all_docs_state: list[Any] | None,
    conversation_state: list[tuple[str, str]] | None,
    detected_mode_state: str | None,
):
    history = chat_history or []
    conversations = conversation_state or []
    all_docs = all_docs_state or []
    document_index = document_index_state or {}
    effective_mode = _resolve_query_mode(mode_choice, detected_mode_state or "general")

    if not user_question or not user_question.strip():
        return history, "", check_model_status(), "請輸入問題。", conversations

    if retriever is None:
        history.append({"role": "assistant", "content": "請先上傳並處理 PDF，完成後再開始提問。"})
        return history, "", check_model_status(), "尚無檢索結果。", conversations

    if model_choice.startswith("Google ADK"):
        try:
            answer, preview = run_adk_agent_answer(
                user_question=user_question.strip(),
                retriever=retriever,
                document_index=document_index,
                all_docs=all_docs,
                conversation_state=conversations,
                detected_mode=detected_mode_state,
            )
            history.append({"role": "user", "content": user_question})
            history.append({"role": "assistant", "content": _to_expandable_answer(answer)})
            conversations.append((user_question, answer))
            return history, "", check_model_status(), preview, conversations
        except Exception as exc:
            history.append({"role": "assistant", "content": f"Google ADK Agent 執行失敗：{_friendly_model_error(exc)}"})
            return history, "", check_model_status(), "ADK 檢索失敗，請確認 Gemini API 與文件狀態。", conversations

    try:
        fallback_note = ""
        context_docs: list[Any] = []
        candidate_hint = ""
        normalized_question = user_question.strip()
        list_query = any(marker in normalized_question for marker in ("有哪些", "有哪一些", "相關分類"))
        lookup_query = is_code_or_lookup_query(normalized_question)
        page_lookup = is_page_query(normalized_question)
        inference_query = _is_general_inference_question(normalized_question)
        summary_query = _is_summary_like_question(normalized_question) and not inference_query
        try:
            llm = _resolve_llm(
                model_choice,
                num_predict_override=_gemma_num_predict_for_query(
                    is_summary=summary_query,
                    is_list_query=list_query,
                    is_lookup_query=lookup_query,
                    is_page_lookup=page_lookup,
                ),
            )
            if page_lookup:
                pages = extract_page_numbers(user_question)
                context_docs = [
                    doc
                    for doc in all_docs
                    if int(doc.metadata.get("page", 0) or 0) in pages
                ]
                if not context_docs:
                    answer = f"文件中找不到你指定的頁碼：{', '.join(f'第 {page} 頁' for page in pages)}。"
                else:
                    candidate_hint = summarize_page_docs(context_docs)
                    if _should_use_candidate_directly(
                        candidate_hint,
                        detected_mode=detected_mode_state,
                        is_summary=summary_query,
                        is_list_query=list_query,
                        is_lookup_query=lookup_query,
                        is_page_lookup=page_lookup,
                    ):
                        answer = candidate_hint
                    else:
                        prompt_docs = _prepare_reasoning_context(
                            user_question,
                            context_docs,
                            is_list_query=list_query,
                            is_lookup_query=lookup_query,
                            is_page_lookup=page_lookup,
                        )
                        prompt = _build_reasoning_prompt(
                            user_question,
                            prompt_docs,
                            document_index,
                            detected_mode_state,
                            candidate_hint,
                        )
                        answer = _invoke_llm_text(llm, prompt) or summarize_page_docs(context_docs)
                        answer = _finalize_hybrid_answer(answer, candidate_hint)
                preview = _format_retrieved_preview(context_docs)
                history.append({"role": "user", "content": user_question})
                history.append({"role": "assistant", "content": _to_expandable_answer(answer)})
                conversations.append((user_question, answer))
                return history, "", check_model_status(), preview, conversations

            if summary_query:
                context_docs = all_docs[: min(len(all_docs), 8)]
                candidate_hint = (
                    answer_with_document_index(user_question, document_index)
                    if _is_index_document_mode(detected_mode_state)
                    else ""
                )
                if model_choice.startswith("Gemma") and not _is_index_document_mode(detected_mode_state):
                    answer = _build_general_summary_fallback(all_docs)
                    preview = _format_retrieved_preview(context_docs)
                    history.append({"role": "user", "content": user_question})
                    history.append({"role": "assistant", "content": _to_expandable_answer(answer)})
                    conversations.append((user_question, answer))
                    return history, "", check_model_status(), preview, conversations
                if _should_use_candidate_directly(
                    candidate_hint,
                    detected_mode=detected_mode_state,
                    is_summary=summary_query,
                    is_list_query=list_query,
                    is_lookup_query=lookup_query,
                    is_page_lookup=page_lookup,
                ):
                    answer = candidate_hint
                else:
                    prompt = (
                        build_summary_prompt(user_question, document_index, all_docs)
                        if _is_index_document_mode(detected_mode_state)
                        else _build_generic_summary_prompt(user_question, all_docs)
                    )
                    if candidate_hint:
                        prompt += f"\n\n可參考的文件結構草稿：\n{candidate_hint}"
                    answer = _invoke_llm_text(llm, prompt)
                    if not _is_index_document_mode(detected_mode_state) and _needs_candidate_rewrite(answer):
                        answer = _build_general_summary_fallback(all_docs)
                    answer = _finalize_hybrid_answer(answer, candidate_hint)
                preview = _format_retrieved_preview(context_docs)
                history.append({"role": "user", "content": user_question})
                history.append({"role": "assistant", "content": _to_expandable_answer(answer)})
                conversations.append((user_question, answer))
                return history, "", check_model_status(), preview, conversations

            if lookup_query:
                context_docs = _select_reasoning_docs(
                    user_question,
                    all_docs,
                    retriever,
                    document_index,
                    top_k=6 if list_query else 4,
                )
                candidate_hint = (
                    answer_with_document_index(user_question, document_index)
                    if _is_index_document_mode(detected_mode_state)
                    else ""
                )

            # Fast-path for literal keyword/phrase lookups only.
            if should_use_literal_lookup(normalized_question):
                exact_matches_all = _find_exact_matches(normalized_question, all_docs)
                if exact_matches_all:
                    context_docs = exact_matches_all
                    if not candidate_hint:
                        candidate_hint = _build_exact_match_answer(user_question, exact_matches_all)

            if not context_docs:
                context_docs = _select_reasoning_docs(
                    user_question,
                    all_docs,
                    retriever,
                    document_index,
                    top_k=5,
                )
            if _should_use_candidate_directly(
                candidate_hint,
                detected_mode=detected_mode_state,
                is_summary=summary_query,
                is_list_query=list_query,
                is_lookup_query=lookup_query,
                is_page_lookup=page_lookup,
            ):
                answer = candidate_hint
            else:
                prompt_docs = _prepare_reasoning_context(
                    user_question,
                    context_docs,
                    is_list_query=list_query,
                    is_lookup_query=lookup_query,
                    is_page_lookup=page_lookup,
                )
                prompt = _build_reasoning_prompt(
                    user_question,
                    prompt_docs,
                    document_index,
                    detected_mode_state,
                    candidate_hint,
                )
                answer = _invoke_llm_text(llm, prompt)
                if (
                    model_choice.startswith("Gemma")
                    and not _is_index_document_mode(detected_mode_state)
                    and inference_query
                    and _needs_candidate_rewrite(answer)
                ):
                    answer = _build_general_inference_fallback(user_question, context_docs) or answer
                answer = _finalize_hybrid_answer(answer, candidate_hint)
        except Exception as primary_exc:
            # If Gemini fails (quota/network), automatically fallback to local Gemma.
            if not model_choice.startswith("Gemma"):
                try:
                    llm = get_gemma_llm(
                        num_predict_override=_gemma_num_predict_for_query(
                            is_summary=summary_query,
                            is_list_query=list_query,
                            is_lookup_query=lookup_query,
                            is_page_lookup=page_lookup,
                        )
                    )
                    if page_lookup:
                        pages = extract_page_numbers(user_question)
                        context_docs = [
                            doc
                            for doc in all_docs
                            if int(doc.metadata.get("page", 0) or 0) in pages
                        ]
                        candidate_hint = summarize_page_docs(context_docs)
                        if _should_use_candidate_directly(
                            candidate_hint,
                            detected_mode=detected_mode_state,
                            is_summary=summary_query,
                            is_list_query=list_query,
                            is_lookup_query=lookup_query,
                            is_page_lookup=page_lookup,
                        ):
                            answer = candidate_hint
                        else:
                            prompt_docs = _prepare_reasoning_context(
                                user_question,
                                context_docs,
                                is_list_query=list_query,
                                is_lookup_query=lookup_query,
                                is_page_lookup=page_lookup,
                            )
                            prompt = _build_reasoning_prompt(
                                user_question,
                                prompt_docs,
                                document_index,
                                detected_mode_state,
                                candidate_hint,
                            )
                            answer = _invoke_llm_text(llm, prompt) or summarize_page_docs(context_docs)
                            answer = _finalize_hybrid_answer(answer, candidate_hint)
                    elif summary_query:
                        context_docs = all_docs[: min(len(all_docs), 8)]
                        candidate_hint = (
                            answer_with_document_index(user_question, document_index)
                            if _is_index_document_mode(detected_mode_state)
                            else ""
                        )
                        if _should_use_candidate_directly(
                            candidate_hint,
                            detected_mode=detected_mode_state,
                            is_summary=summary_query,
                            is_list_query=list_query,
                            is_lookup_query=lookup_query,
                            is_page_lookup=page_lookup,
                        ):
                            answer = candidate_hint
                        else:
                            prompt = (
                                build_summary_prompt(user_question, document_index, all_docs)
                                if _is_index_document_mode(detected_mode_state)
                                else _build_generic_summary_prompt(user_question, all_docs)
                            )
                            if candidate_hint:
                                prompt += f"\n\n可參考的文件結構草稿：\n{candidate_hint}"
                            answer = _invoke_llm_text(llm, prompt)
                            if not _is_index_document_mode(detected_mode_state) and _needs_candidate_rewrite(answer):
                                answer = _build_general_summary_fallback(all_docs)
                            answer = _finalize_hybrid_answer(answer, candidate_hint)
                    else:
                        exact_matches_all = []
                        candidate_hint = (
                            answer_with_document_index(user_question, document_index)
                            if _is_index_document_mode(detected_mode_state)
                            else ""
                        )
                        if should_use_literal_lookup(normalized_question):
                            exact_matches_all = _find_exact_matches(normalized_question, all_docs)
                        if exact_matches_all:
                            context_docs = exact_matches_all
                            if not candidate_hint:
                                candidate_hint = _build_exact_match_answer(user_question, exact_matches_all)
                        else:
                            context_docs = _select_reasoning_docs(
                                user_question,
                                all_docs,
                                retriever,
                                document_index,
                                top_k=6 if list_query else 4,
                            )
                        if _should_use_candidate_directly(
                            candidate_hint,
                            detected_mode=detected_mode_state,
                            is_summary=summary_query,
                            is_list_query=list_query,
                            is_lookup_query=lookup_query,
                            is_page_lookup=page_lookup,
                        ):
                            answer = candidate_hint
                        else:
                            prompt_docs = _prepare_reasoning_context(
                                user_question,
                                context_docs,
                                is_list_query=list_query,
                                is_lookup_query=lookup_query,
                                is_page_lookup=page_lookup,
                            )
                            prompt = _build_reasoning_prompt(
                                user_question,
                                prompt_docs,
                                document_index,
                                detected_mode_state,
                                candidate_hint,
                            )
                            answer = _invoke_llm_text(llm, prompt)
                            if (
                                not _is_index_document_mode(detected_mode_state)
                                and inference_query
                                and _needs_candidate_rewrite(answer)
                            ):
                                answer = _build_general_inference_fallback(user_question, context_docs) or answer
                            answer = _finalize_hybrid_answer(answer, candidate_hint)
                    fallback_note = (
                        "（Gemini 失敗，已自動改用 Gemma 本地模型）\n"
                        f"原因：{_friendly_model_error(primary_exc)}"
                    )
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"{_friendly_model_error(primary_exc)}；且 Gemma 也失敗：{_friendly_model_error(fallback_exc)}"
                    ) from fallback_exc
            else:
                raise RuntimeError(_friendly_model_error(primary_exc)) from primary_exc

        answer = answer.strip() or "根據文件內容，我找不到相關資訊"
        source_docs = context_docs

        match_candidates = list(source_docs)
        if ENABLE_BROAD_SEARCH and hasattr(retriever, "vectorstore"):
            try:
                broader_docs = retriever.vectorstore.similarity_search(
                    user_question.strip(),
                    k=max(TOP_K, BROAD_SEARCH_K),
                )
                if broader_docs:
                    match_candidates.extend(broader_docs)
            except Exception:
                pass

        exact_matches = _find_exact_matches(user_question.strip(), match_candidates)
        should_override_with_exact = (
            should_use_literal_lookup(user_question.strip())
            and not list_query
            and not lookup_query
            and not page_lookup
            and not summary_query
            and _needs_candidate_rewrite(answer)
        )
        if exact_matches and should_override_with_exact:
            answer = _build_exact_match_answer(user_question, exact_matches)
        preview = _format_retrieved_preview(source_docs)

        pages = sorted(
            {
                f"第 {doc.metadata.get('page', '?')} 頁"
                for doc in source_docs
                if doc.metadata.get("page") is not None
            }
        )
        if pages:
            answer = f"{answer}\n\n參考來源：{'、'.join(pages)}"
        if fallback_note:
            answer = f"{answer}\n\n{fallback_note}"

        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": _to_expandable_answer(answer)})
        conversations.append((user_question, answer))
        return history, "", check_model_status(), preview, conversations
    except Exception as exc:
        history.append({"role": "assistant", "content": f"處理問題時發生錯誤：{exc}"})
        return history, "", check_model_status(), "檢索失敗，請確認模型或文件狀態。", conversations


def clear_chat():
    return [], "", "尚無檢索結果。", []


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PDF 智能問答系統") as demo:
        retriever_state = gr.State(value=None)
        document_index_state = gr.State(value={})
        all_docs_state = gr.State(value=[])
        conversation_state = gr.State(value=[])
        detected_mode_state = gr.State(value="general")

        gr.Markdown("# 📄 PDF 智能問答系統")
        gr.Markdown("支援 Gemma 4 本地模型（Ollama）、Gemini 雲端模型，以及嵌入式 Google ADK Agent。")

        with gr.Row():
            with gr.Column(scale=1):
                uploader = gr.File(
                    label="上傳 PDF（可多檔）",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )
                model_choice = gr.Dropdown(
                    label="模型選擇",
                    choices=["Gemma 4 (本地)", "Gemini (雲端)", "Google ADK Agent (嵌入式)"],
                    value="Gemma 4 (本地)",
                )
                mode_choice = gr.Dropdown(
                    label="問答模式",
                    choices=["自動", "通用推理模式"],
                    value="通用推理模式",
                )
                process_status = gr.Markdown("尚未載入 PDF。")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="對話區", height=480)
                with gr.Row():
                    user_input = gr.Textbox(
                        label="輸入問題",
                        placeholder="請輸入你想問 PDF 的問題...",
                        interactive=False,
                    )
                    send_btn = gr.Button("送出", variant="primary", interactive=False)
                clear_btn = gr.Button("清除對話")

        model_status = gr.Markdown(check_model_status())
        doc_mode_status = gr.Markdown(_format_doc_mode_status("general"))

        with gr.Accordion("最近一次檢索段落預覽", open=False):
            preview = gr.Markdown("尚無檢索結果。")

        uploader.change(
            fn=process_uploaded_pdfs,
            inputs=[uploader],
            outputs=[
                retriever_state,
                document_index_state,
                process_status,
                model_status,
                doc_mode_status,
                preview,
                user_input,
                send_btn,
                all_docs_state,
                conversation_state,
                detected_mode_state,
            ],
        )

        send_btn.click(
            fn=ask_question,
            inputs=[user_input, chatbot, retriever_state, document_index_state, model_choice, mode_choice, all_docs_state, conversation_state, detected_mode_state],
            outputs=[chatbot, user_input, model_status, preview, conversation_state],
        )
        user_input.submit(
            fn=ask_question,
            inputs=[user_input, chatbot, retriever_state, document_index_state, model_choice, mode_choice, all_docs_state, conversation_state, detected_mode_state],
            outputs=[chatbot, user_input, model_status, preview, conversation_state],
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, user_input, preview, conversation_state],
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
