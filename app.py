from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter
import re
import time

import gradio as gr
import requests
from langchain_core.documents import Document

from config import (
    BM25_CANDIDATE_K,
    BROAD_SEARCH_K,
    COMPLEX_REASONING_MIN_CONTEXT_DOCS,
    COMPLEX_REASONING_MIN_QUESTION_LEN,
    COMPLEX_REASONING_PASSES,
    COMPLEX_RETRIEVAL_TOP_K_BOOST,
    CROSS_ENCODER_MODEL,
    CROSS_ENCODER_TOP_K,
    ENABLE_BM25_RETRIEVAL,
    ENABLE_CROSS_ENCODER_RERANK,
    ENABLE_BROAD_SEARCH,
    ENABLE_RETRIEVAL_CACHE,
    GEMINI_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_GEMMA_MODEL,
    OLLAMA_QWEN_MODEL,
    OLLAMA_NUM_PREDICT,
    QUERY_VARIANT_COUNT,
    RETRIEVAL_CACHE_SIZE,
    RETRIEVAL_PROFILE,
    RRF_RANK_CONSTANT,
    TOP_K,
)
from models.gemma_local import get_ollama_llm
from models.gemini_cloud import get_gemini_llm
from rag.csv_loader import load_csv_and_split
from rag.pdf_loader import load_and_split
from rag.web_loader import load_web_and_split
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
)
from rag.vector_store import create_vector_store, get_retriever, get_vector_store_stats

_MODEL_STATUS_CACHE: str | None = None
_MODEL_STATUS_LAST_TS = 0.0
_MODEL_STATUS_TTL_SECONDS = 5.0
_EXPANDABLE_THRESHOLD = 350
_CSV_HIDDEN_FIELDS = {"source_pdf", "note", "page", "pdf", "pdf_path", "pdf_source"}
_RETRIEVAL_CACHE: dict[str, list[Any]] = {}
_RETRIEVAL_CACHE_ORDER: list[str] = []
_CROSS_ENCODER_MODEL = None
_CROSS_ENCODER_LOAD_FAILED = False


def _friendly_model_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "google adk agent" in lowered or "google_adk" in lowered:
        return message
    if "429" in message or "resource_exhausted" in lowered or "quota" in lowered:
        return "Gemini 配額不足（429 RESOURCE_EXHAUSTED）。請稍後再試，或先改用本地 Ollama 模型。"
    if "winerror 10061" in lowered or "connecterror" in lowered or "failed to establish a new connection" in lowered:
        return "Gemini 連線失敗（可能是代理設定或網路限制）。已建議改用本地 Ollama 模型。"
    if "api key" in lowered or "permission" in lowered or "unauthorized" in lowered:
        return "Gemini 驗證失敗，請確認 API Key 是否正確且可用。"
    return f"模型呼叫失敗：{message}"


def _is_local_ollama_choice(model_choice: str) -> bool:
    return model_choice.startswith("Gemma") or model_choice.startswith("Qwen")


def _ollama_model_for_choice(model_choice: str) -> str:
    if model_choice.startswith("Qwen"):
        return OLLAMA_QWEN_MODEL
    return OLLAMA_GEMMA_MODEL


def _resolve_llm(model_choice: str, num_predict_override: int | None = None):
    if _is_local_ollama_choice(model_choice):
        return get_ollama_llm(
            model_name=_ollama_model_for_choice(model_choice),
            num_predict_override=num_predict_override,
        )
    return get_gemini_llm()


def check_model_status() -> str:
    """Return markdown status for Ollama and Gemini availability."""
    global _MODEL_STATUS_CACHE
    global _MODEL_STATUS_LAST_TS

    now = time.time()
    if _MODEL_STATUS_CACHE is not None and (now - _MODEL_STATUS_LAST_TS) < _MODEL_STATUS_TTL_SECONDS:
        return _MODEL_STATUS_CACHE

    ollama_status = "不可用"
    local_model_status = "未檢測"
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.ok:
            models = {m.get("name", "") for m in response.json().get("models", []) if isinstance(m, dict)}
            ollama_status = "已連線"
            gemma_ok = OLLAMA_GEMMA_MODEL in models
            qwen_ok = OLLAMA_QWEN_MODEL in models
            gemma_text = f"Gemma `{OLLAMA_GEMMA_MODEL}` {'可用' if gemma_ok else '缺少'}"
            qwen_text = f"Qwen `{OLLAMA_QWEN_MODEL}` {'可用' if qwen_ok else '缺少'}"
            local_model_status = f"{gemma_text}；{qwen_text}"
        else:
            ollama_status = "服務異常"
            local_model_status = "模型清單讀取失敗"
    except requests.RequestException:
        ollama_status = "未連線（請確認 ollama serve）"
        local_model_status = "無法檢測"

    gemini_status = "API Key 已設定（未即時檢測配額/連線）" if GEMINI_API_KEY.strip() else "未設定 GEMINI_API_KEY"
    status = (
        "### 模型狀態\n"
        f"- Ollama：{ollama_status}\n"
        f"- 本地模型：{local_model_status}\n"
        f"- Gemini：{gemini_status}"
    )
    _MODEL_STATUS_CACHE = status
    _MODEL_STATUS_LAST_TS = now
    return status


def _detect_source_type(doc: Any) -> str:
    source_type = str(doc.metadata.get("source_type", "") or "").strip().lower()
    if source_type:
        return source_type
    source = str(doc.metadata.get("source", "") or "").strip().lower()
    if source.startswith(("http://", "https://")):
        return "web"
    if source.endswith(".csv"):
        return "csv"
    return "pdf"


def _format_position_label(doc: Any) -> str:
    position = doc.metadata.get("page", "?")
    source_type = _detect_source_type(doc)
    if source_type == "csv":
        return f"第 {position} 列"
    if source_type == "web":
        return f"第 {position} 段"
    return f"第 {position} 頁"


def _filtered_csv_fields(doc: Any) -> dict[str, str]:
    fields = _extract_csv_fields(doc)
    filtered: dict[str, str] = {}
    for key, value in fields.items():
        key_norm = key.strip().lower()
        if key_norm in _CSV_HIDDEN_FIELDS:
            continue
        filtered[key] = value
    return filtered


def _format_csv_doc_content(doc: Any) -> str:
    fields = _filtered_csv_fields(doc)
    ordered_keys = ["code", "title", "title_excerpt", "definition"]
    lines: list[str] = []
    used: set[str] = set()

    for key in ordered_keys:
        for actual_key, value in fields.items():
            if actual_key.strip().lower() != key:
                continue
            if not value:
                continue
            lines.append(f"{actual_key}: {_clean_ocr_text(value)}")
            used.add(actual_key)

    for actual_key, value in fields.items():
        if actual_key in used:
            continue
        if not value:
            continue
        lines.append(f"{actual_key}: {_clean_ocr_text(value)}")

    return "\n".join(lines).strip()


def _summarize_position_docs(position_docs: list[Any], *, position_kind: str = "auto") -> str:
    if not position_docs:
        return "找不到你指定的位置內容。"

    labels: list[str] = []
    blocks: list[str] = []
    for doc in position_docs:
        label = _format_position_label(doc)
        if label not in labels:
            labels.append(label)
        if _detect_source_type(doc) == "csv":
            content = _format_csv_doc_content(doc)
        else:
            content = _clean_ocr_text(str(getattr(doc, "page_content", "") or ""))
        blocks.append(f"[{label}]\n{content}")

    if position_kind == "row":
        head = f"你指定的位置是：{'、'.join(labels)}（CSV 以列為單位）。"
    elif position_kind == "page":
        head = f"你指定的位置是：{'、'.join(labels)}。"
    else:
        head = f"你指定的位置是：{'、'.join(labels)}。"
    return f"{head}\n位置完整內容：\n" + "\n\n".join(blocks)


def _format_retrieved_preview(source_docs: list[Any]) -> str:
    if not source_docs:
        return "尚無檢索結果。"

    lines: list[str] = []
    for idx, doc in enumerate(source_docs[:4], start=1):
        source = doc.metadata.get("source", "未知檔案")
        position_label = _format_position_label(doc)
        if _detect_source_type(doc) == "csv":
            content = _format_csv_doc_content(doc).replace("\n", " ")
        else:
            content = (doc.page_content or "").strip().replace("\n", " ")
        if len(content) > 220:
            content = content[:220] + "..."
        lines.append(f"{idx}. [{source} - {position_label}] {content}")

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


def _extract_csv_fields(doc: Any) -> dict[str, str]:
    fields: dict[str, str] = {}
    content = str(getattr(doc, "page_content", "") or "")
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_text = key.strip()
        value_text = value.strip()
        if key_text:
            fields[key_text] = value_text
    return fields


def _clean_ocr_text(text: str) -> str:
    cleaned = (text or "").replace("\u3000", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。；：！？、）】」』])", "", cleaned)
    cleaned = re.sub(r"(?<=[（【「『])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    return cleaned


def _question_terms_for_match(question: str) -> list[str]:
    normalized = _normalize_text(question)
    cleaned = normalized
    for marker in ("是什麼", "是甚麼", "什麼", "請問", "介紹", "說明", "內容", "定義", "嗎", "呢"):
        cleaned = cleaned.replace(marker, " ")
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", cleaned)

    terms: list[str] = []
    terms.extend([term for term in cleaned.split() if len(term) >= 2])
    terms.extend(re.findall(r"[\u4e00-\u9fff]{2,}", cleaned))
    terms.extend(re.findall(r"[a-z0-9][a-z0-9_-]{1,}", cleaned))

    stop_terms = {"請", "一下", "可以", "the", "what", "about", "row"}
    deduped: list[str] = []
    for term in terms:
        if term in stop_terms:
            continue
        if term not in deduped:
            deduped.append(term)
    return deduped


def _try_csv_direct_answer(user_question: str, all_docs: list[Any]) -> tuple[str, list[Any]] | None:
    csv_docs = [doc for doc in all_docs if _detect_source_type(doc) == "csv"]
    if not csv_docs:
        return None

    normalized_question = _normalize_text(user_question)
    terms = _question_terms_for_match(user_question)
    phrase = normalized_question
    for marker in ("是什麼", "是甚麼", "什麼", "請問", "介紹", "說明", "內容", "定義", "嗎", "呢", "？", "?"):
        phrase = phrase.replace(marker, " ")
    phrase = re.sub(r"\s+", " ", phrase).strip()
    code_match = re.search(r"\b[0-9A-Z]{4,6}\b", user_question.upper())
    code = code_match.group(0) if code_match else ""

    scored: list[tuple[int, Any, dict[str, str]]] = []
    for doc in csv_docs:
        fields = _extract_csv_fields(doc)
        text = _normalize_text(str(getattr(doc, "page_content", "") or ""))
        score = 0
        if code and code in text.upper():
            score += 6
        for term in terms:
            if term in text:
                score += 2
        if phrase and phrase in text:
            score += 5
        if phrase:
            overlap = len({ch for ch in phrase if ch.strip() and ch in text})
            if overlap >= 3:
                score += min(6, overlap)
        if normalized_question and normalized_question in text:
            score += 4
        if score > 0:
            scored.append((score, doc, fields))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    top_score = scored[0][0]
    best = [item for item in scored if item[0] >= max(2, top_score - 1)][:2]
    selected_docs = [item[1] for item in best]
    top_fields = best[0][2]

    code_text = top_fields.get("code", "")
    title_text = _clean_ocr_text(top_fields.get("title_excerpt", "") or top_fields.get("title", ""))
    definition_text = _clean_ocr_text(top_fields.get("definition", "") or "")
    if not title_text and definition_text:
        title_text = definition_text
    position = _format_position_label(best[0][1])

    lines = ["已在 CSV 中找到相符資料。"]
    if code_text:
        lines.append(f"- 代碼：{code_text}")
    if title_text:
        lines.append(f"- 標題：{title_text}")
    if definition_text and definition_text != title_text:
        lines.append(f"- 定義：{definition_text}")
    lines.append(f"- 來源：{position}")

    if len(best) > 1:
        alt_fields = best[1][2]
        alt_code = alt_fields.get("code", "").strip()
        alt_title = (
            alt_fields.get("title_excerpt", "")
            or alt_fields.get("title", "")
            or alt_fields.get("definition", "")
        )
        alt_title = _clean_ocr_text(alt_title)
        if alt_code or alt_title:
            lines.append("- 可能相關：")
            lines.append(f"  {alt_code} {alt_title}".strip())

    return "\n".join(lines), selected_docs


def _extract_row_numbers(user_question: str) -> list[int]:
    text = user_question.strip().lower()
    numbers: set[int] = set()
    patterns = (
        r"第\s*(\d+)\s*(?:列|行)",
        r"\brow\s*(\d+)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                value = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if value > 0:
                numbers.add(value)
    return sorted(numbers)


def _is_row_query(user_question: str) -> bool:
    return bool(_extract_row_numbers(user_question))


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


def _is_list_query(user_question: str) -> bool:
    normalized = _normalize_text(user_question)
    marker_hits = (
        "有哪些",
        "有哪一些",
        "請列出",
        "列出",
        "清單",
        "全部",
        "所有",
        "what are",
        "list all",
    )
    if any(marker in normalized for marker in marker_hits):
        return True
    return bool(
        re.search(r"(哪些|哪幾).*(代碼|條目|項目|分類|類型|章節)", normalized)
        or re.search(r"(codes|items|categories)", normalized)
    )


def _build_context_summary(context_docs: list[Any], *, max_docs: int = 5, max_chars: int = 140) -> str:
    if not context_docs:
        return ""
    lines: list[str] = []
    for doc in context_docs[:max_docs]:
        position = _format_position_label(doc)
        source = str(doc.metadata.get("source", "") or "未知來源")
        snippet = _clean_ocr_text(str(getattr(doc, "page_content", "") or ""))
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."
        lines.append(f"- [{source} - {position}] {snippet}")
    return "\n".join(lines)


def _compact_text(text: str, *, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


def _build_conversation_context(
    conversations: list[tuple[str, str]] | None,
    *,
    max_turns: int = 4,
    max_chars_per_item: int = 180,
) -> str:
    items = conversations or []
    if not items:
        return ""

    lines: list[str] = []
    for idx, (question, answer) in enumerate(items[-max_turns:], start=1):
        q = _compact_text(question, max_chars=max_chars_per_item)
        a = _compact_text(answer, max_chars=max_chars_per_item)
        if not q and not a:
            continue
        lines.append(f"{idx}. Q: {q}")
        lines.append(f"   A: {a}")
    return "\n".join(lines).strip()


def _invoke_llm_text(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = str(getattr(response, "content", "") or "").strip()
    if content:
        return content
    return ""


def _is_complex_reasoning_query(
    user_question: str,
    *,
    resolved_query_mode: str,
    context_doc_count: int,
) -> bool:
    if resolved_query_mode in {"page_lookup", "lookup"}:
        return False

    normalized = _normalize_text(user_question)
    multi_step_markers = (
        "比較",
        "差異",
        "推論",
        "為什麼",
        "原因",
        "如何",
        "策略",
        "評估",
        "綜合",
        "分析",
        "tradeoff",
        "trade-off",
        "compare",
        "reason",
        "analyze",
        "analysis",
    )
    if any(marker in normalized for marker in multi_step_markers):
        return True

    if len(normalized) >= max(8, COMPLEX_REASONING_MIN_QUESTION_LEN):
        return True

    if context_doc_count >= max(2, COMPLEX_REASONING_MIN_CONTEXT_DOCS):
        return True

    return resolved_query_mode in {"inference", "summary", "list"}


def _build_refine_prompt(user_question: str, draft_answer: str, context_docs: list[Any]) -> str:
    context_summary = _build_context_summary(context_docs, max_docs=6, max_chars=170)
    if not context_summary:
        context_summary = "- 目前沒有額外上下文摘要。"
    return (
        "你是文件問答助手。請把下方『初稿答案』改寫成更完整、可驗證、結構化的最終答案。\n"
        "規則：\n"
        "1) 先輸出「可確認事實」，再輸出「推論」。\n"
        "2) 不可捏造文件中沒有的細節。\n"
        "3) 若證據不足，要明確標示不確定處。\n"
        "4) 回答用繁體中文。\n\n"
        f"問題：{user_question.strip()}\n\n"
        f"初稿答案：\n{(draft_answer or '').strip()}\n\n"
        f"上下文摘要：\n{context_summary}\n"
    )


def _invoke_reasoning_answer(
    llm,
    prompt: str,
    *,
    user_question: str,
    resolved_query_mode: str,
    context_docs: list[Any],
) -> str:
    answer = _invoke_llm_text(llm, prompt)
    passes = max(1, COMPLEX_REASONING_PASSES)
    if passes <= 1:
        return answer

    if not _is_complex_reasoning_query(
        user_question,
        resolved_query_mode=resolved_query_mode,
        context_doc_count=len(context_docs),
    ):
        return answer

    for _ in range(1, passes):
        refine_prompt = _build_refine_prompt(user_question, answer, context_docs)
        refined = _invoke_llm_text(llm, refine_prompt)
        if refined.strip():
            answer = refined.strip()
    return answer


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
    locations: set[str] = set()
    for doc in docs[:3]:
        locations.add(_format_position_label(doc))
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
    if locations:
        lines.append(f"\n參考來源： {', '.join(sorted(locations))}")
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


def _doc_identity(doc: Any) -> tuple[str, int, str]:
    source = str(doc.metadata.get("source", "") or "")
    page = int(doc.metadata.get("page", 0) or 0)
    head = str(getattr(doc, "page_content", "") or "")[:120]
    return source, page, head


def _configured_retrieval_profile() -> str:
    profile = (RETRIEVAL_PROFILE or "auto").strip().lower()
    if profile not in {"precise", "balanced", "explore", "auto"}:
        return "auto"
    return profile


def _resolve_retrieval_profile(
    user_question: str,
    *,
    all_docs: list[Any],
    is_summary: bool,
    is_list_query: bool,
    is_lookup_query: bool,
    is_page_lookup: bool,
    is_inference_query: bool,
) -> str:
    configured = _configured_retrieval_profile()
    if configured != "auto":
        return configured

    if is_page_lookup:
        return "precise"
    if is_summary or is_list_query or is_inference_query:
        return "explore"

    source_types = {_detect_source_type(doc) for doc in all_docs}
    if len(source_types) >= 2:
        return "explore"

    question = user_question.strip()
    terms = _query_terms(question)
    if len(question) >= 28 and len(terms) <= 3 and not is_lookup_query:
        return "explore"
    if is_lookup_query:
        return "balanced"
    return "balanced"


def _variant_limit(profile: str) -> int:
    base = max(2, QUERY_VARIANT_COUNT)
    if profile == "precise":
        return min(base, 4)
    if profile == "explore":
        return min(max(base, 6), 10)
    return min(base, 8)


def _selection_top_k(base_top_k: int, profile: str) -> int:
    if profile == "precise":
        return max(3, base_top_k)
    if profile == "explore":
        return max(8, base_top_k + 2)
    return max(6, base_top_k + 1)


def _cache_get(key: str) -> list[Any] | None:
    if not ENABLE_RETRIEVAL_CACHE:
        return None
    docs = _RETRIEVAL_CACHE.get(key)
    if docs is None:
        return None
    if key in _RETRIEVAL_CACHE_ORDER:
        _RETRIEVAL_CACHE_ORDER.remove(key)
    _RETRIEVAL_CACHE_ORDER.append(key)
    return docs


def _cache_set(key: str, docs: list[Any]) -> None:
    if not ENABLE_RETRIEVAL_CACHE:
        return
    _RETRIEVAL_CACHE[key] = docs
    if key in _RETRIEVAL_CACHE_ORDER:
        _RETRIEVAL_CACHE_ORDER.remove(key)
    _RETRIEVAL_CACHE_ORDER.append(key)
    while len(_RETRIEVAL_CACHE_ORDER) > max(8, RETRIEVAL_CACHE_SIZE):
        oldest = _RETRIEVAL_CACHE_ORDER.pop(0)
        _RETRIEVAL_CACHE.pop(oldest, None)


def _build_retrieval_cache_key(user_question: str, docs: list[Any], top_k: int, profile: str) -> str:
    head = [_doc_identity(doc) for doc in docs[:20]]
    tail = [_doc_identity(doc) for doc in docs[-5:]] if len(docs) > 20 else []
    fingerprint = f"{len(docs)}|{head}|{tail}"
    question = _normalize_text(user_question)
    return f"{profile}|k={top_k}|q={question}|d={fingerprint}"


def _query_terms(question: str) -> list[str]:
    cleaned = _normalize_text(question)
    terms = [term for term in re.split(r"\s+", cleaned) if len(term) >= 2]
    extra_zh = re.findall(r"[\u4e00-\u9fff]{2,}", cleaned)
    extra_code = re.findall(r"\b[0-9a-z][0-9a-z._-]{1,}\b", cleaned)
    deduped: list[str] = []
    for term in terms + extra_zh + extra_code:
        if term not in deduped:
            deduped.append(term)
    return deduped


def _bm25_tokens(text: str) -> list[str]:
    tokens = _query_terms(text)
    if tokens:
        return tokens
    normalized = _normalize_text(text)
    latin = re.findall(r"[a-z0-9][a-z0-9._-]{1,}", normalized)
    zh = re.findall(r"[一-鿿]{1,2}", normalized)
    combined = [item for item in latin + zh if item.strip()]
    if combined:
        return combined
    return [normalized] if normalized else []


def _bm25_rank_docs(user_question: str, docs: list[Any], *, top_k: int) -> list[Any]:
    if not ENABLE_BM25_RETRIEVAL or not docs:
        return []
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return []

    corpus_tokens: list[list[str]] = []
    valid_docs: list[Any] = []
    for doc in docs:
        content = str(getattr(doc, "page_content", "") or "")
        tokens = _bm25_tokens(content)
        if not tokens:
            continue
        corpus_tokens.append(tokens)
        valid_docs.append(doc)

    if not corpus_tokens:
        return []

    query_tokens = _bm25_tokens(user_question)
    if not query_tokens:
        return []

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)
    ranked_pairs = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    picked: list[Any] = []
    for index, score in ranked_pairs:
        if float(score) <= 0:
            continue
        picked.append(valid_docs[index])
        if len(picked) >= max(1, top_k):
            break
    return picked


def _get_cross_encoder_model():
    global _CROSS_ENCODER_MODEL
    global _CROSS_ENCODER_LOAD_FAILED

    if _CROSS_ENCODER_MODEL is not None:
        return _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_LOAD_FAILED:
        return None

    try:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER_MODEL = CrossEncoder(CROSS_ENCODER_MODEL)
        return _CROSS_ENCODER_MODEL
    except Exception:
        _CROSS_ENCODER_LOAD_FAILED = True
        return None


def _cross_encoder_rerank(user_question: str, docs: list[Any], *, top_k: int) -> list[Any]:
    if not ENABLE_CROSS_ENCODER_RERANK or not docs:
        return docs
    model = _get_cross_encoder_model()
    if model is None:
        return docs

    capped_docs = docs[: max(top_k, 1)]
    pairs = [
        [user_question, str(getattr(doc, "page_content", "") or "")[:1200]]
        for doc in capped_docs
    ]

    try:
        scores = model.predict(pairs)
    except Exception:
        return docs

    ranked_pairs = sorted(
        zip(capped_docs, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    reranked_head = [doc for doc, _ in ranked_pairs]
    tail = [doc for doc in docs if all(_doc_identity(doc) != _doc_identity(head) for head in reranked_head)]
    return reranked_head + tail


def _decompose_complex_question(question: str, *, max_parts: int = 4) -> list[str]:
    text = re.sub(r"\s+", " ", (question or "")).strip()
    if not text:
        return []

    fragments = [part.strip() for part in re.split(r"[。！？!?；;\n]+", text) if part.strip()]
    if len(fragments) <= 1:
        fragments = [
            part.strip()
            for part in re.split(r"(?:以及|並且|同時|另外|還有|and|then)", text, flags=re.I)
            if part.strip()
        ]

    if len(fragments) <= 1:
        return []

    parts: list[str] = []
    for part in fragments:
        cleaned = re.sub(r"\s+", " ", part).strip("，、 ")
        if len(cleaned) < 6:
            continue
        if cleaned not in parts:
            parts.append(cleaned)
        if len(parts) >= max_parts:
            break
    return parts


def _build_query_variants(user_question: str, profile: str) -> list[str]:
    question = user_question.strip()
    if not question:
        return []

    variants: list[str] = [question]
    terms = _query_terms(question)
    code_match = re.search(r"\b[0-9A-Z]{3,8}\b", question.upper())
    if code_match:
        code = code_match.group(0)
        variants.append(code)
        variants.append(f"{code} 定義")
        variants.append(f"{code} 是什麼")

    if terms:
        variants.append(" ".join(terms[:6]))
        variants.append(" ".join(terms[:3]))

    for sub_question in _decompose_complex_question(question, max_parts=4):
        variants.append(sub_question)
        sub_terms = _query_terms(sub_question)
        if sub_terms:
            variants.append(" ".join(sub_terms[:5]))

    # Defensive expansion for unusual prompts.
    variants.append(f"請找出與這題最相關的內容：{question}")
    variants.append(f"最接近這題意圖的段落：{question}")

    unique: list[str] = []
    for item in variants:
        text = re.sub(r"\s+", " ", item).strip()
        if len(text) < 2:
            continue
        if text not in unique:
            unique.append(text)
    return unique[:_variant_limit(profile)]


def _retriever_search(retriever, query: str, k: int) -> list[Any]:
    try:
        if hasattr(retriever, "vectorstore"):
            return retriever.vectorstore.similarity_search(query, k=max(1, k))
    except Exception:
        pass

    try:
        if hasattr(retriever, "invoke"):
            result = retriever.invoke(query)
            if isinstance(result, list):
                return result[:k]
    except Exception:
        pass

    try:
        if hasattr(retriever, "get_relevant_documents"):
            result = retriever.get_relevant_documents(query)
            if isinstance(result, list):
                return result[:k]
    except Exception:
        pass
    return []


def _rrf_fuse_ranked_lists(ranked_lists: list[list[Any]], rank_constant: int = RRF_RANK_CONSTANT) -> list[Any]:
    scores: dict[tuple[str, int, str], float] = {}
    doc_by_key: dict[tuple[str, int, str], Any] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_identity(doc)
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + (1.0 / (rank_constant + rank))

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_by_key[key] for key, _ in ordered]


def _question_fit_score(user_question: str, doc: Any) -> float:
    question_terms = _query_terms(user_question)
    content = _normalize_text(str(getattr(doc, "page_content", "") or ""))
    if not content:
        return 0.0

    score = 0.0
    for term in question_terms:
        if term in content:
            score += 1.0

    code_match = re.search(r"\b[0-9A-Z]{3,8}\b", user_question.upper())
    if code_match and code_match.group(0).lower() in content:
        score += 3.0

    # Prefer chunks with richer details for odd/open-ended questions.
    score += min(len(content), 600) / 1200.0
    return score


def _rerank_with_question_fit(user_question: str, docs: list[Any]) -> list[Any]:
    ranked = sorted(
        docs,
        key=lambda doc: (_question_fit_score(user_question, doc),),
        reverse=True,
    )
    return ranked


def _diversify_docs(docs: list[Any], limit: int) -> list[Any]:
    if not docs:
        return []
    picked: list[Any] = []
    by_source_count: dict[str, int] = {}
    for pass_cap in (1, 2, 99):
        for doc in docs:
            if len(picked) >= limit:
                return picked
            key = _doc_identity(doc)
            if any(_doc_identity(existing) == key for existing in picked):
                continue
            source = str(doc.metadata.get("source", "") or "")
            count = by_source_count.get(source, 0)
            if count >= pass_cap:
                continue
            picked.append(doc)
            by_source_count[source] = count + 1
    return picked[:limit]


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
    retrieval_profile: str | None = None,
) -> list[Any]:
    profile = retrieval_profile or _configured_retrieval_profile()
    if profile == "auto":
        profile = "balanced"
    effective_top_k = _selection_top_k(top_k, profile=profile)
    cache_key = _build_retrieval_cache_key(user_question, all_docs, effective_top_k, profile)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached[:effective_top_k]

    selected: list[Any] = []
    ranked_lists: list[list[Any]] = []

    heading_matches = find_heading_matches(user_question, document_index)[:8]
    matched_pages = {int(item.get("page", 0) or 0) for item in heading_matches if item.get("page")}
    if matched_pages:
        heading_docs = [
            doc for doc in all_docs if int(doc.metadata.get("page", 0) or 0) in matched_pages
        ]
        selected.extend(heading_docs)
        ranked_lists.append(heading_docs)

    section_matches = find_section_matches(user_question, document_index)[:3]
    section_titles = {item.get("title", "") for item in section_matches if item.get("title")}
    if section_titles:
        section_pages = {
            int(item.get("page", 0) or 0)
            for item in document_index.get("headings", [])
            if item.get("section", "") in section_titles
        }
        section_docs = [
            doc for doc in all_docs if int(doc.metadata.get("page", 0) or 0) in section_pages
        ]
        selected.extend(section_docs)
        ranked_lists.append(section_docs)

    lookup_terms = extract_lookup_terms(user_question)
    for term in lookup_terms[:4]:
        normalized_term = _normalize_text(term)
        if not normalized_term or len(normalized_term) < 2:
            continue
        literal_docs = [
            doc
            for doc in all_docs
            if normalized_term in _normalize_text(str(getattr(doc, "page_content", "") or ""))
        ]
        if literal_docs:
            selected.extend(literal_docs[: max(effective_top_k + 2, 8)])
            ranked_lists.append(literal_docs[: max(effective_top_k + 2, 8)])

    bm25_docs = _bm25_rank_docs(
        user_question,
        all_docs,
        top_k=max(4, min(BM25_CANDIDATE_K, effective_top_k + 8)),
    )
    if bm25_docs:
        ranked_lists.append(bm25_docs)
        selected.extend(bm25_docs)

    query_variants = _build_query_variants(user_question, profile=profile)
    for query in query_variants:
        docs_for_query: list[Any] = []
        hybrid_items = hybrid_retrieve(query, all_docs, retriever, top_k=max(effective_top_k + 2, 8))
        docs_for_query.extend(
            item.document if hasattr(item, "document") else item
            for item in hybrid_items
        )
        docs_for_query.extend(_retriever_search(retriever, query, k=max(effective_top_k + 2, 8)))
        docs_for_query = _dedupe_docs(docs_for_query)
        if docs_for_query:
            ranked_lists.append(docs_for_query[: max(effective_top_k + 1, 6)])
            selected.extend(docs_for_query[: max(effective_top_k + 1, 6)])

    if ranked_lists:
        fused_docs = _rrf_fuse_ranked_lists(ranked_lists, rank_constant=RRF_RANK_CONSTANT)
    else:
        fused_docs = _dedupe_docs(selected)

    reranked = _rerank_with_question_fit(user_question, _dedupe_docs(fused_docs + selected))
    reranked = _cross_encoder_rerank(
        user_question,
        reranked,
        top_k=max(effective_top_k + 2, CROSS_ENCODER_TOP_K),
    )
    final_docs = _diversify_docs(reranked, limit=max(effective_top_k, 1))
    _cache_set(cache_key, final_docs)
    return final_docs


def _build_reasoning_prompt(
    user_question: str,
    context_docs: list[Any],
    document_index: dict[str, Any],
    detected_mode: str | None,
    candidate_hint: str = "",
    *,
    resolved_query_mode: str = "general",
    is_local_model: bool = False,
    conversation_context: str = "",
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
    for doc in context_docs[:12]:
        position_label = _format_position_label(doc)
        text = (doc.page_content or "").strip()
        source = str(doc.metadata.get("source", "") or "未知來源")
        context_blocks.append(f"[{source} - {position_label}]\n{text}")
    joined_context = "\n\n".join(context_blocks)

    hint_block = candidate_hint.strip() or candidate_block
    context_summary = _build_context_summary(context_docs) if resolved_query_mode in {"inference", "general"} else ""
    summary_block = f"\n\n上下文摘要（先讀這段再回答）：\n{context_summary}" if context_summary else ""
    history_block = f"\n\n先前對話重點：\n{conversation_context.strip()}" if conversation_context.strip() else ""

    if resolved_query_mode == "list":
        return (
            "你是 PDF 文件問答助手，請優先根據提供段落回答，使用繁體中文。\n"
            "這是一題『完整列舉』問題。你必須列出段落中所有符合條件的條目，不可以只回其中一筆。\n"
            "若段落中出現多個代碼與名稱，請全部整理成清單，每行包含代碼、名稱、頁碼與所屬分類。\n"
            "你可以參考候選答案草稿，但最終答案必須由你根據文件段落整理後輸出。\n"
            "若證據不足，請先列出最接近的候選代碼/關鍵字，再說明缺少哪些證據。\n\n"
            f"候選答案草稿：\n{hint_block}\n\n"
            f"候選資訊：\n{candidate_block}\n\n"
            f"文件段落：\n{joined_context}{summary_block}{history_block}\n\n"
            f"問題：{question}"
        )
    if resolved_query_mode == "lookup":
        return (
            "你是 PDF 文件問答助手，請優先根據提供段落回答，使用繁體中文。\n"
            "這是一題『單一條目說明』問題。請優先回答名稱、代碼、所屬分類、頁碼與定義。\n"
            "你可以參考候選答案草稿，但最終答案必須由你根據文件段落整理後輸出。\n"
            "若段落已經有對應條目，不要直接回答找不到。\n"
            "若欄位證據不足，請標記該欄位缺少依據，並附上最接近的關鍵字。\n\n"
            f"候選答案草稿：\n{hint_block}\n\n"
            f"候選資訊：\n{candidate_block}\n\n"
            f"文件段落：\n{joined_context}{summary_block}{history_block}\n\n"
            f"問題：{question}"
        )
    if _is_index_document_mode(detected_mode):
        return (
            build_llm_prompt(user_question, context_docs, document_index)
            + f"\n\n候選答案草稿：\n{hint_block}\n\n候選資訊：\n{candidate_block}{history_block}"
        )

    if is_local_model:
        return (
            "### 角色：專業文檔分析官\n"
            "### 任務：\n"
            "1. 閱讀以下提供的【參考片段】，這些片段可能不完整。\n"
            "2. 即使片段中沒有直接答案，也請從上下文推論最有可能的相關資訊。\n"
            "3. 如果真的無法直接回答，請列出最接近的關鍵字與依據段落。\n"
            "4. 回答時請先列出可確認事實，再補充推論與不確定處。\n\n"
            f"【參考片段開始】\n{joined_context}\n【參考片段結束】"
            f"{summary_block}{history_block}\n\n"
            f"### 使用者問題：{question}\n"
            "### 回答指引：請優先條列式呈現事實，最後再進行總結。"
        )

    return (
        "你是多來源文件問答助手（PDF/CSV/Web），請使用繁體中文，優先根據提供的內容回答。\n"
        "請先抽取直接證據，再給結論；若問題較跳躍，先提供最接近的可驗證資訊。\n"
        "若需要推論，請明確標示「可確認」與「推論」兩部分。\n"
        "若證據不足，請列出最接近關鍵字，不要只回覆找不到。\n"
        "不要把文件誤判成固定主題，除非段落中真的有相關內容。\n\n"
        f"文件段落：\n{joined_context}{summary_block}{history_block}\n\n"
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
    inference_query = _is_general_inference_question(user_question) and not is_list_query and not is_lookup_query
    if is_page_lookup:
        return _compress_context_docs(user_question, context_docs, max_docs=3, max_chars=900)
    if is_list_query:
        return _compress_context_docs(user_question, context_docs, max_docs=10, max_chars=520)
    if inference_query:
        return _compress_context_docs(user_question, context_docs, max_docs=8, max_chars=520)
    if is_lookup_query:
        return _compress_context_docs(user_question, context_docs, max_docs=7, max_chars=420)
    return _compress_context_docs(user_question, context_docs, max_docs=8, max_chars=460)


def _build_evidence_grounded_fallback(user_question: str, context_docs: list[Any], all_docs: list[Any]) -> str:
    docs = context_docs or all_docs[:4]
    if not docs:
        return "目前沒有可用文件內容，因此無法回答這個問題。"

    lines = [
        "這題在文件中沒有直接的標準答案，我先提供最接近的可驗證資訊：",
    ]
    for doc in docs[:4]:
        source = str(doc.metadata.get("source", "") or "未知來源")
        position = _format_position_label(doc)
        snippet = _clean_ocr_text((doc.page_content or "").strip())
        if len(snippet) > 170:
            snippet = snippet[:170] + "..."
        lines.append(f"- [{source} - {position}] {snippet}")
    lines.append("若你要，我可以再用這些證據改寫成「更像你問題語氣」的答案。")
    return "\n".join(lines)


def _profile_debug_line(resolved_profile: str) -> str:
    configured = _configured_retrieval_profile()
    if configured == "auto":
        return f"檢索策略：`auto -> {resolved_profile}`"
    return f"檢索策略：`{resolved_profile}`"


def _preview_with_profile(preview_text: str, resolved_profile: str) -> str:
    line = _profile_debug_line(resolved_profile)
    body = (preview_text or "").strip()
    if not body:
        return line
    return f"{line}\n\n{body}"


def _is_index_document_mode(detected_mode: str | None) -> bool:
    return (detected_mode or "").strip().lower() == "index"


def _build_generic_summary_prompt(
    user_question: str,
    all_docs: list[Any],
    *,
    conversation_context: str = "",
) -> str:
    intro_blocks: list[str] = []
    for doc in all_docs[: min(len(all_docs), 6)]:
        position_label = _format_position_label(doc)
        text = (doc.page_content or "").strip().replace("\n", " ")
        if len(text) > 360:
            text = text[:360] + "..."
        intro_blocks.append(f"[{position_label}] {text}")

    context = "\n\n".join(intro_blocks) if intro_blocks else "目前沒有可用的文件內容。"
    history_block = (
        f"\n\n先前對話重點：\n{conversation_context.strip()}"
        if conversation_context.strip()
        else ""
    )
    return (
        "你是 PDF 文件摘要助手，請使用繁體中文，只根據提供的文件內容回答。\n"
        "如果使用者在問這份文件的大意、摘要、總結、概述或在說什麼，"
        "請整理主題與重點，不要把文件硬套成 ICD 或其他特定領域。\n"
        "若文件內容不足以完整概括，就明確說明目前能確認的內容。\n\n"
        f"文件片段：\n{context}{history_block}\n\n"
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
        or _is_row_query(question)
        or is_code_or_lookup_query(question)
        or should_use_literal_lookup(question)
    )


def _resolve_query_mode(mode_choice: str, detected_mode: str, user_question: str) -> str:
    question = user_question.strip()
    row_lookup = _is_row_query(question)
    page_lookup = is_page_query(question) or row_lookup
    summary_query = _is_summary_like_question(question) and not _is_general_inference_question(question)
    list_query = _is_list_query(question)
    inference_query = _is_general_inference_question(question) and not list_query
    lookup_query = is_code_or_lookup_query(question) or should_use_literal_lookup(question)

    if page_lookup:
        return "page_lookup"
    if summary_query:
        return "summary"
    if list_query:
        return "list"
    if lookup_query:
        return "lookup"
    if inference_query:
        return "inference"

    if mode_choice == "通用推理模式" and detected_mode in {"general", "web", "csv"}:
        return "inference"
    return "general"


def _format_doc_mode_status(detected_mode: str) -> str:
    profile = _configured_retrieval_profile()
    if detected_mode == "index":
        return f"### 文件模式\n- 自動判定：`索引型 PDF`\n- 目前策略：`混合式`\n- 檢索檔位：`{profile}`\n- 說明：先讓模型根據檢索內容生成；若模型失手，再退回結構化候選答案。"
    if detected_mode == "csv":
        return f"### 文件模式\n- 自動判定：`CSV`\n- 目前策略：`列優先`\n- 檢索檔位：`{profile}`\n- 說明：CSV 來源與定位單位使用「列」，不使用頁碼。"
    if detected_mode == "web":
        return f"### 文件模式\n- 自動判定：`Web`\n- 目前策略：`段落檢索`\n- 檢索檔位：`{profile}`\n- 說明：網頁內容已切段，定位單位使用「段」。"
    return f"### 文件模式\n- 自動判定：`一般 PDF`\n- 目前策略：`混合式`\n- 檢索檔位：`{profile}`\n- 說明：保留 RAG + LLM 推理，必要時以結構化候選答案補強。"



def _load_documents(file_path: str) -> list[Document]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return load_and_split(file_path)
    if suffix == ".csv":
        return load_csv_and_split(file_path)
    raise ValueError(f"目前不支援的檔案格式：{suffix}")


def _parse_web_urls(web_urls_text: str | None) -> list[str]:
    if not web_urls_text or not web_urls_text.strip():
        return []
    candidates = re.split(r"[\r\n,]+", web_urls_text)
    urls: list[str] = []
    for item in candidates:
        value = item.strip()
        if not value:
            continue
        if not value.lower().startswith(("http://", "https://")):
            continue
        if value not in urls:
            urls.append(value)
    return urls


def _source_status_label(source: str) -> str:
    text = source.strip()
    if text.startswith(("http://", "https://")) and len(text) > 72:
        return text[:72] + "..."
    return Path(text).name if Path(text).name else text


def _web_fetch_label(docs: list[Document]) -> str:
    if not docs:
        return "requests"
    method = str(docs[0].metadata.get("fetch_method", "") or "").strip().lower()
    return method or "requests"


def process_uploaded_sources(files: list[str] | None, web_urls_text: str | None = None):
    global _RETRIEVAL_CACHE
    global _RETRIEVAL_CACHE_ORDER

    web_urls = _parse_web_urls(web_urls_text)
    if not files and not web_urls:
        return (
            None,
            {},
            "請先上傳至少一份 PDF/CSV，或輸入至少一個網頁 URL。",
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
    used_sources: list[str] = []
    failed_sources: list[str] = []

    try:
        for file_path in files or []:
            try:
                docs = _load_documents(file_path)
                all_docs.extend(docs)
                used_sources.append(_source_status_label(file_path))
                for doc in docs:
                    source = str(doc.metadata.get("source", ""))
                    page = int(doc.metadata.get("page", 0))
                    if source and page > 0:
                        unique_pages.add((source, page))
            except Exception as exc:
                failed_sources.append(f"{_source_status_label(file_path)}：{exc}")

        for url in web_urls:
            try:
                docs = load_web_and_split(url)
                all_docs.extend(docs)
                used_sources.append(f"{_source_status_label(url)} (web:{_web_fetch_label(docs)})")
                for doc in docs:
                    source = str(doc.metadata.get("source", ""))
                    page = int(doc.metadata.get("page", 0))
                    if source and page > 0:
                        unique_pages.add((source, page))
            except Exception as exc:
                failed_sources.append(f"{_source_status_label(url)}：{exc}")

        if not all_docs:
            failure_text = "；".join(failed_sources) if failed_sources else "沒有可用來源。"
            return (
                None,
                {},
                f"來源處理失敗：{failure_text}",
                check_model_status(),
                _format_doc_mode_status("general"),
                "尚無檢索結果。",
                gr.update(interactive=False),
                gr.update(interactive=False),
                [],
                [],
                "general",
            )

        has_pdf_docs = any(_detect_source_type(doc) == "pdf" for doc in all_docs)
        has_csv_docs = any(_detect_source_type(doc) == "csv" for doc in all_docs)
        has_web_docs = any(_detect_source_type(doc) == "web" for doc in all_docs)
        if has_pdf_docs:
            document_index = build_document_index(all_docs)
            detected_mode = detect_document_mode(document_index, all_docs)
        elif has_csv_docs:
            document_index = {}
            detected_mode = "csv"
        elif has_web_docs:
            document_index = {}
            detected_mode = "web"
        else:
            document_index = {}
            detected_mode = "general"
        vector_store = create_vector_store(all_docs)
        retriever = get_retriever(vector_store, top_k=TOP_K)
        vector_stats = get_vector_store_stats()
        _RETRIEVAL_CACHE = {}
        _RETRIEVAL_CACHE_ORDER = []

        status = (
            f"已載入 {len(unique_pages)} 個來源位置（PDF 以頁、CSV 以列、Web 以段），切割為 {len(all_docs)} 個段落。"
            f"\n來源：{', '.join(used_sources)}"
        )
        status += (
            "\n向量庫："
            f"cache_hit={vector_stats.get('cache_hits', 0)}，"
            f"reuse={vector_stats.get('reused_collections', 0)}，"
            f"build={vector_stats.get('built_collections', 0)}，"
            f"cleanup_removed={vector_stats.get('cleanup_removed_collections', 0)}，"
            f"fallback={vector_stats.get('fallback_builds', 0)}"
        )
        last_collection = str(vector_stats.get("last_collection", "") or "")
        if last_collection:
            status += f"\n目前 collection：{last_collection}"
        if failed_sources:
            status += "\n以下來源載入失敗（已略過）：\n- " + "\n- ".join(failed_sources[:8])
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
            f"來源處理失敗：{exc}",
            check_model_status(),
            _format_doc_mode_status("general"),
            "尚無檢索結果。",
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
            [],
            "general",
        )


def process_uploaded_pdfs(files: list[str] | None):
    return process_uploaded_sources(files=files, web_urls_text=None)


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
    if not user_question or not user_question.strip():
        return history, "", check_model_status(), "請輸入問題。", conversations

    if retriever is None:
        history.append({"role": "assistant", "content": "請先上傳並處理來源（PDF / CSV / Web），完成後再開始提問。"})
        return history, "", check_model_status(), "尚無檢索結果。", conversations

    try:
        fallback_note = ""
        context_docs: list[Any] = []
        candidate_hint = ""
        normalized_question = user_question.strip()
        has_csv_docs = any(_detect_source_type(doc) == "csv" for doc in all_docs)
        has_pdf_docs = any(_detect_source_type(doc) == "pdf" for doc in all_docs)
        if has_csv_docs and not has_pdf_docs:
            detected_mode_state = "csv"

        resolved_query_mode = _resolve_query_mode(mode_choice, detected_mode_state or "general", normalized_question)
        list_query = resolved_query_mode == "list"
        lookup_query = resolved_query_mode == "lookup"
        row_lookup = _is_row_query(normalized_question)
        page_lookup = resolved_query_mode == "page_lookup" or row_lookup
        inference_query = resolved_query_mode == "inference"
        summary_query = resolved_query_mode == "summary"
        conversation_context = _build_conversation_context(conversations)

        retrieval_top_k = 7
        if resolved_query_mode == "list":
            retrieval_top_k = 12
        elif resolved_query_mode == "inference":
            retrieval_top_k = 10
        elif resolved_query_mode == "summary":
            retrieval_top_k = 10
        elif resolved_query_mode == "lookup":
            retrieval_top_k = 8

        if _is_complex_reasoning_query(
            normalized_question,
            resolved_query_mode=resolved_query_mode,
            context_doc_count=0,
        ):
            retrieval_top_k += max(0, COMPLEX_RETRIEVAL_TOP_K_BOOST)

        resolved_profile = _resolve_retrieval_profile(
            normalized_question,
            all_docs=all_docs,
            is_summary=summary_query,
            is_list_query=list_query,
            is_lookup_query=lookup_query,
            is_page_lookup=page_lookup,
            is_inference_query=inference_query,
        )

        csv_direct = _try_csv_direct_answer(user_question, all_docs)
        if csv_direct is not None and not summary_query and not page_lookup:
            answer, context_docs = csv_direct
            preview = _preview_with_profile(_format_retrieved_preview(context_docs), resolved_profile)
            history.append({"role": "user", "content": user_question})
            history.append({"role": "assistant", "content": _to_expandable_answer(answer)})
            conversations.append((user_question, answer))
            return history, "", check_model_status(), preview, conversations

        try:
            csv_speed_override = (
                min(OLLAMA_NUM_PREDICT, 128)
                if _is_local_ollama_choice(model_choice) and has_csv_docs and not summary_query
                else None
            )
            llm = _resolve_llm(
                model_choice,
                num_predict_override=(
                    csv_speed_override
                    if csv_speed_override is not None
                    else _gemma_num_predict_for_query(
                        is_summary=summary_query,
                        is_list_query=list_query,
                        is_lookup_query=lookup_query,
                        is_page_lookup=page_lookup,
                    )
                ),
            )
            if page_lookup:
                positions = _extract_row_numbers(user_question) if row_lookup else extract_page_numbers(user_question)
                if not positions:
                    unit_hint = "列號" if row_lookup else "頁碼"
                    answer = f"請在問題中指定有效的{unit_hint}（例如：第 3 {'列' if row_lookup else '頁'}）。"
                    preview = _preview_with_profile(_format_retrieved_preview(context_docs), resolved_profile)
                    history.append({"role": "user", "content": user_question})
                    history.append({"role": "assistant", "content": _to_expandable_answer(answer)})
                    conversations.append((user_question, answer))
                    return history, "", check_model_status(), preview, conversations
                context_docs = []
                for doc in all_docs:
                    source_type = _detect_source_type(doc)
                    if row_lookup and source_type != "csv":
                        continue
                    if (not row_lookup) and source_type == "csv":
                        continue
                    if int(doc.metadata.get("page", 0) or 0) in positions:
                        context_docs.append(doc)
                if not context_docs:
                    unit = "列" if row_lookup else "頁"
                    answer = f"文件中找不到你指定的位置：{', '.join(f'第 {position} {unit}' for position in positions)}。"
                else:
                    candidate_hint = _summarize_position_docs(
                        context_docs,
                        position_kind="row" if row_lookup else "page",
                    )
                    if row_lookup:
                        answer = candidate_hint
                        preview = _preview_with_profile(_format_retrieved_preview(context_docs), resolved_profile)
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
                            resolved_query_mode=resolved_query_mode,
                            is_local_model=_is_local_ollama_choice(model_choice),
                            conversation_context=conversation_context,
                        )
                        answer = (
                            _invoke_reasoning_answer(
                                llm,
                                prompt,
                                user_question=user_question,
                                resolved_query_mode=resolved_query_mode,
                                context_docs=prompt_docs,
                            )
                            or candidate_hint
                        )
                        answer = _finalize_hybrid_answer(answer, candidate_hint)
                preview = _preview_with_profile(_format_retrieved_preview(context_docs), resolved_profile)
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
                if _is_local_ollama_choice(model_choice) and not _is_index_document_mode(detected_mode_state):
                    answer = _build_general_summary_fallback(all_docs)
                    preview = _preview_with_profile(_format_retrieved_preview(context_docs), resolved_profile)
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
                        else _build_generic_summary_prompt(
                            user_question,
                            all_docs,
                            conversation_context=conversation_context,
                        )
                    )
                    if candidate_hint:
                        prompt += f"\n\n可參考的文件結構草稿：\n{candidate_hint}"
                    answer = _invoke_reasoning_answer(
                        llm,
                        prompt,
                        user_question=user_question,
                        resolved_query_mode=resolved_query_mode,
                        context_docs=context_docs,
                    )
                    if not _is_index_document_mode(detected_mode_state) and _needs_candidate_rewrite(answer):
                        answer = _build_general_summary_fallback(all_docs)
                    answer = _finalize_hybrid_answer(answer, candidate_hint)
                preview = _preview_with_profile(_format_retrieved_preview(context_docs), resolved_profile)
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
                    top_k=max(5, retrieval_top_k),
                    retrieval_profile=resolved_profile,
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
                    top_k=retrieval_top_k,
                    retrieval_profile=resolved_profile,
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
                    resolved_query_mode=resolved_query_mode,
                    is_local_model=_is_local_ollama_choice(model_choice),
                    conversation_context=conversation_context,
                )
                answer = _invoke_reasoning_answer(
                    llm,
                    prompt,
                    user_question=user_question,
                    resolved_query_mode=resolved_query_mode,
                    context_docs=prompt_docs,
                )
                if (
                    _is_local_ollama_choice(model_choice)
                    and not _is_index_document_mode(detected_mode_state)
                    and inference_query
                    and _needs_candidate_rewrite(answer)
                ):
                    answer = _build_general_inference_fallback(user_question, context_docs) or answer
                answer = _finalize_hybrid_answer(answer, candidate_hint)
        except Exception as primary_exc:
            # If Gemini fails (quota/network), automatically fallback to local Ollama model.
            if not _is_local_ollama_choice(model_choice):
                try:
                    csv_speed_override = (
                        min(OLLAMA_NUM_PREDICT, 128)
                        if has_csv_docs and not summary_query
                        else None
                    )
                    llm = get_ollama_llm(
                        model_name=_ollama_model_for_choice("Gemma 4 (本地)"),
                        num_predict_override=(
                            csv_speed_override
                            if csv_speed_override is not None
                            else _gemma_num_predict_for_query(
                                is_summary=summary_query,
                                is_list_query=list_query,
                                is_lookup_query=lookup_query,
                                is_page_lookup=page_lookup,
                            )
                        )
                    )
                    if page_lookup:
                        positions = _extract_row_numbers(user_question) if row_lookup else extract_page_numbers(user_question)
                        if not positions:
                            unit_hint = "列號" if row_lookup else "頁碼"
                            answer = f"請在問題中指定有效的{unit_hint}（例如：第 3 {'列' if row_lookup else '頁'}）。"
                            preview = _preview_with_profile(_format_retrieved_preview(context_docs), resolved_profile)
                            history.append({"role": "user", "content": user_question})
                            history.append({"role": "assistant", "content": _to_expandable_answer(answer)})
                            conversations.append((user_question, answer))
                            return history, "", check_model_status(), preview, conversations
                        context_docs = []
                        for doc in all_docs:
                            source_type = _detect_source_type(doc)
                            if row_lookup and source_type != "csv":
                                continue
                            if (not row_lookup) and source_type == "csv":
                                continue
                            if int(doc.metadata.get("page", 0) or 0) in positions:
                                context_docs.append(doc)
                        candidate_hint = _summarize_position_docs(
                            context_docs,
                            position_kind="row" if row_lookup else "page",
                        )
                        if row_lookup:
                            answer = candidate_hint
                            preview = _preview_with_profile(_format_retrieved_preview(context_docs), resolved_profile)
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
                                resolved_query_mode=resolved_query_mode,
                                is_local_model=_is_local_ollama_choice(model_choice),
                                conversation_context=conversation_context,
                            )
                            answer = (
                                _invoke_reasoning_answer(
                                    llm,
                                    prompt,
                                    user_question=user_question,
                                    resolved_query_mode=resolved_query_mode,
                                    context_docs=prompt_docs,
                                )
                                or candidate_hint
                            )
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
                                else _build_generic_summary_prompt(
                                    user_question,
                                    all_docs,
                                    conversation_context=conversation_context,
                                )
                            )
                            if candidate_hint:
                                prompt += f"\n\n可參考的文件結構草稿：\n{candidate_hint}"
                            answer = _invoke_reasoning_answer(
                                llm,
                                prompt,
                                user_question=user_question,
                                resolved_query_mode=resolved_query_mode,
                                context_docs=context_docs,
                            )
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
                                top_k=max(5, retrieval_top_k),
                                retrieval_profile=resolved_profile,
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
                                resolved_query_mode=resolved_query_mode,
                                is_local_model=_is_local_ollama_choice(model_choice),
                                conversation_context=conversation_context,
                            )
                            answer = _invoke_reasoning_answer(
                                llm,
                                prompt,
                                user_question=user_question,
                                resolved_query_mode=resolved_query_mode,
                                context_docs=prompt_docs,
                            )
                            if (
                                not _is_index_document_mode(detected_mode_state)
                                and inference_query
                                and _needs_candidate_rewrite(answer)
                            ):
                                answer = _build_general_inference_fallback(user_question, context_docs) or answer
                            answer = _finalize_hybrid_answer(answer, candidate_hint)
                    fallback_note = (
                        "（Gemini 失敗，已自動改用本地 Ollama 模型）\n"
                        f"原因：{_friendly_model_error(primary_exc)}"
                    )
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"{_friendly_model_error(primary_exc)}；且本地 Ollama 模型也失敗：{_friendly_model_error(fallback_exc)}"
                    ) from fallback_exc
            else:
                raise RuntimeError(_friendly_model_error(primary_exc)) from primary_exc

        if _needs_candidate_rewrite(answer) and not page_lookup:
            answer = _build_evidence_grounded_fallback(user_question, context_docs, all_docs)
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
        preview = _preview_with_profile(_format_retrieved_preview(source_docs), resolved_profile)

        locations = sorted(
            {
                _format_position_label(doc)
                for doc in source_docs
                if doc.metadata.get("page") is not None
            }
        )
        if locations:
            answer = f"{answer}\n\n參考來源：{'、'.join(locations)}"
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
    with gr.Blocks(title="文件智能問答系統") as demo:
        retriever_state = gr.State(value=None)
        document_index_state = gr.State(value={})
        all_docs_state = gr.State(value=[])
        conversation_state = gr.State(value=[])
        detected_mode_state = gr.State(value="general")

        gr.Markdown("# 📄 文件智能問答系統")
        gr.Markdown("支援 PDF / CSV / Web 內容載入，並可使用 Gemma 4 / Qwen3 14B 本地模型（Ollama）與 Gemini 雲端模型。")

        with gr.Row():
            with gr.Column(scale=1):
                uploader = gr.File(
                    label="上傳檔案（PDF / CSV，可多檔）",
                    file_count="multiple",
                    file_types=[".pdf", ".csv"],
                    type="filepath",
                )
                web_url_input = gr.Textbox(
                    label="網頁 URL（可多行）",
                    placeholder="每行一個網址，例如：https://example.com/article",
                    lines=3,
                )
                load_sources_btn = gr.Button("載入檔案 / 網頁來源", variant="secondary")
                model_choice = gr.Dropdown(
                    label="模型選擇",
                    choices=["Gemma 4 (本地)", "Qwen3 14B (本地)", "Gemini (雲端)"],
                    value="Gemma 4 (本地)",
                )
                mode_choice = gr.Dropdown(
                    label="問答模式",
                    choices=["自動", "通用推理模式"],
                    value="通用推理模式",
                )
                process_status = gr.Markdown("尚未載入檔案或網頁。")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="對話區", height=480)
                with gr.Row():
                    user_input = gr.Textbox(
                        label="輸入問題",
                        placeholder="請輸入你想問檔案內容的問題...",
                        interactive=False,
                    )
                    send_btn = gr.Button("送出", variant="primary", interactive=False)
                clear_btn = gr.Button("清除對話")

        model_status = gr.Markdown(check_model_status())
        doc_mode_status = gr.Markdown(_format_doc_mode_status("general"))

        with gr.Accordion("最近一次檢索段落預覽", open=False):
            preview = gr.Markdown("尚無檢索結果。")

        uploader.change(
            fn=process_uploaded_sources,
            inputs=[uploader, web_url_input],
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

        load_sources_btn.click(
            fn=process_uploaded_sources,
            inputs=[uploader, web_url_input],
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
