from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from rag.pdf_loader import load_and_split
from rag.query_engine import (
    answer_with_document_index,
    build_document_index,
    detect_document_mode,
    extract_page_numbers,
    find_heading_matches,
    hybrid_retrieve,
    is_page_query,
    summarize_page_docs,
)
from rag.vector_store import create_vector_store, get_retriever

try:
    from google.adk.agents.context import Context as ToolContext
except Exception:  # pragma: no cover
    ToolContext = Any


@lru_cache(maxsize=8)
def _load_pdf_runtime(pdf_path: str) -> dict[str, Any]:
    docs = load_and_split(pdf_path)
    doc_index = build_document_index(docs)
    detected_mode = detect_document_mode(doc_index, docs)
    retriever = get_retriever(create_vector_store(docs), top_k=4)
    return {
        "docs": docs,
        "doc_index": doc_index,
        "detected_mode": detected_mode,
        "retriever": retriever,
    }


def _validate_pdf_path(pdf_path: str) -> str:
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"找不到 PDF 檔案：{pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"目前只支援 PDF 檔案：{pdf_path}")
    return str(path)


def _resolve_active_pdf(pdf_path: str | None, tool_context: ToolContext | None = None) -> str:
    if pdf_path and str(pdf_path).strip():
        resolved = _validate_pdf_path(str(pdf_path).strip())
        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["active_pdf_path"] = resolved
        return resolved

    if tool_context is not None and hasattr(tool_context, "state"):
        active = tool_context.state.get("active_pdf_path")
        if active:
            return _validate_pdf_path(str(active))

    raise ValueError("尚未指定 PDF 路徑。請先提供 pdf_path，或先使用 set_active_pdf。")


def _build_general_summary(docs: list[Any]) -> str:
    page_count = len({int(doc.metadata.get("page", 0) or 0) for doc in docs if doc.metadata.get("page")})
    sample_docs = docs[: min(len(docs), 6)]
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
    return "\n".join(lines_out)


def summarize_pdf(pdf_path: str) -> str:
    """Summarize what a PDF is about. Good for 'what is this document about' style questions."""
    resolved = _validate_pdf_path(pdf_path)
    runtime = _load_pdf_runtime(resolved)
    if runtime["detected_mode"] == "index":
        return answer_with_document_index("請整理這份文件的大綱。", runtime["doc_index"])
    return _build_general_summary(runtime["docs"])


def set_active_pdf(pdf_path: str, tool_context: ToolContext | None = None) -> str:
    """Set the active PDF for the current ADK session so later questions can omit pdf_path."""
    resolved = _resolve_active_pdf(pdf_path, tool_context)
    return f"已設定目前的 PDF：{resolved}"


def get_active_pdf(tool_context: ToolContext | None = None) -> str:
    """Return the active PDF path for the current ADK session."""
    if tool_context is not None and hasattr(tool_context, "state"):
        active = tool_context.state.get("active_pdf_path")
        if active:
            return f"目前使用中的 PDF：{active}"
    return "目前尚未設定 active PDF。"


def search_pdf(query: str, pdf_path: str | None = None, top_k: int = 4, tool_context: ToolContext | None = None) -> str:
    """Search a PDF and return the most relevant passages with page numbers."""
    resolved = _resolve_active_pdf(pdf_path, tool_context)
    runtime = _load_pdf_runtime(resolved)
    scored = hybrid_retrieve(query.strip(), runtime["docs"], runtime["retriever"], top_k=max(1, min(top_k, 6)))
    docs = [item.document if hasattr(item, "document") else item for item in scored]
    if not docs:
        return "找不到相關段落。"

    blocks: list[str] = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        text = (doc.page_content or "").strip()
        blocks.append(f"[第 {page} 頁]\n{text}")
    return "\n\n".join(blocks)


def lookup_pdf_index(question: str, pdf_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Use structured index lookup when the PDF is index-like, e.g. code/category/page questions."""
    resolved = _resolve_active_pdf(pdf_path, tool_context)
    runtime = _load_pdf_runtime(resolved)
    if runtime["detected_mode"] != "index":
        return "這份文件不是索引型 PDF，請改用 search_pdf 或 summarize_pdf。"
    answer = answer_with_document_index(question.strip(), runtime["doc_index"])
    return answer or "索引中找不到明確答案。"


def get_pdf_page(page_number: int, pdf_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Read the full content of a specific PDF page."""
    resolved = _resolve_active_pdf(pdf_path, tool_context)
    runtime = _load_pdf_runtime(resolved)
    docs = [doc for doc in runtime["docs"] if int(doc.metadata.get("page", 0) or 0) == int(page_number)]
    if not docs:
        return f"文件中找不到第 {page_number} 頁。"
    return summarize_page_docs(docs)


def answer_pdf_question(question: str, pdf_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Return structured evidence for a PDF question. Useful when an ADK agent needs tool-backed reasoning."""
    resolved = _resolve_active_pdf(pdf_path, tool_context)
    runtime = _load_pdf_runtime(resolved)
    question = question.strip()

    if is_page_query(question):
        pages = extract_page_numbers(question)
        if pages:
            return get_pdf_page(resolved, pages[0])

    if runtime["detected_mode"] == "index":
        structured = answer_with_document_index(question, runtime["doc_index"])
        if structured:
            return structured

    heading_matches = find_heading_matches(question, runtime["doc_index"])[:3]
    if heading_matches:
        page_numbers = {int(item.get("page", 0) or 0) for item in heading_matches if item.get("page")}
        docs = [
            doc for doc in runtime["docs"]
            if int(doc.metadata.get("page", 0) or 0) in page_numbers
        ]
        if docs:
            return "\n\n".join(f"[第 {doc.metadata.get('page', '?')} 頁]\n{doc.page_content}" for doc in docs[:3])

    return search_pdf(question, pdf_path=resolved, top_k=4)
