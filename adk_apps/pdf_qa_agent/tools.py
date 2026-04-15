from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADK_ROOT = PROJECT_ROOT / "adk_apps"
ADK_AGENT_ROOT = ADK_ROOT / "pdf_qa_agent"


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


def _normalize_filename(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"\.pdf$", "", normalized)
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)
    return normalized


def _normalize_ascii_filename(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"\.pdf$", "", normalized)
    normalized = re.sub(r"[^0-9a-z]+", "", normalized)
    return normalized


def _looks_like_placeholder_pdf_name(value: str) -> bool:
    norm = _normalize_ascii_filename(value)
    placeholders = {
        "thisdocument",
        "document",
        "uploadedfile",
        "attachment",
        "file",
        "pdf",
    }
    return norm in placeholders


def _pdf_search_roots() -> list[Path]:
    return [
        Path.cwd(),
        PROJECT_ROOT,
        ADK_ROOT,
        ADK_ROOT / ".adk" / "artifacts",
        ADK_AGENT_ROOT / ".adk" / "artifacts",
    ]


def _discover_pdf_files() -> list[Path]:
    files: dict[str, Path] = {}
    skip_dirs = {".git", "venv", "__pycache__", "chroma_db"}

    for root in _pdf_search_roots():
        if not root.exists():
            continue

        direct = list(root.glob("*.pdf"))
        for p in direct:
            if p.is_file():
                files[str(p.resolve())] = p.resolve()

        # Artifact folders need recursive scan, but keep it bounded.
        if "artifacts" in root.parts:
            for p in root.rglob("*.pdf"):
                if p.is_file():
                    files[str(p.resolve())] = p.resolve()
            continue

        for child in root.iterdir():
            if not child.is_dir() or child.name in skip_dirs:
                continue
            for p in child.glob("*.pdf"):
                if p.is_file():
                    files[str(p.resolve())] = p.resolve()

    return sorted(files.values(), key=lambda p: str(p))


def _artifact_roots() -> list[Path]:
    return [
        ADK_ROOT / ".adk" / "artifacts",
        ADK_AGENT_ROOT / ".adk" / "artifacts",
    ]


def _discover_artifact_pdfs() -> list[Path]:
    files: dict[str, Path] = {}
    for root in _artifact_roots():
        if not root.exists():
            continue
        for p in root.rglob("*.pdf"):
            if p.is_file():
                files[str(p.resolve())] = p.resolve()
    return sorted(files.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _latest_pdf_file() -> Path | None:
    pdfs = _discover_pdf_files()
    if not pdfs:
        return None
    return max(pdfs, key=lambda p: p.stat().st_mtime)


def _extract_pdf_hints_from_context(tool_context: ToolContext | None) -> list[str]:
    if tool_context is None:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _push(text: str) -> None:
        for match in re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff ()]+\.pdf", text, flags=re.I):
            key = match.strip()
            if key and key not in seen:
                seen.add(key)
                candidates.append(key)

    # Best-effort scan of commonly available context fields.
    for attr in ("state", "session", "request", "invocation", "user_content", "new_message"):
        value = getattr(tool_context, attr, None)
        if value is not None:
            _push(str(value))

    _push(str(tool_context))
    return candidates


def _match_pdf_from_name(pdf_hint: str) -> Path | None:
    hint = (pdf_hint or "").strip().strip('"').strip("'")
    if not hint:
        return None

    hint_path = Path(hint).expanduser()
    candidate_paths = []
    if hint_path.is_absolute():
        candidate_paths.append(hint_path)
    else:
        candidate_paths.extend([
            Path.cwd() / hint_path,
            PROJECT_ROOT / hint_path,
            ADK_ROOT / hint_path,
        ])

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".pdf":
            return candidate.resolve()

    pdf_files = _discover_pdf_files()
    if not pdf_files:
        return None

    # Exact basename first.
    basename = Path(hint).name
    for p in pdf_files:
        if p.name == basename:
            return p

    normalized_hint = _normalize_filename(basename or hint)
    ascii_hint = _normalize_ascii_filename(basename or hint)
    if not normalized_hint and not ascii_hint:
        return None

    scored: list[tuple[float, Path]] = []
    for p in pdf_files:
        normalized_name = _normalize_filename(p.name)
        ascii_name = _normalize_ascii_filename(p.name)
        if not normalized_name and not ascii_name:
            continue

        if normalized_hint and normalized_name and normalized_hint == normalized_name:
            return p
        if ascii_hint and ascii_name and ascii_hint == ascii_name:
            return p

        if normalized_hint and normalized_name and (normalized_hint in normalized_name or normalized_name in normalized_hint):
            score = 0.95 + min(len(normalized_hint), len(normalized_name)) / 1000
            scored.append((score, p))
            continue

        if ascii_hint and ascii_name and (ascii_hint in ascii_name or ascii_name in ascii_hint):
            score = 0.94 + min(len(ascii_hint), len(ascii_name)) / 1000
            scored.append((score, p))
            continue

        ratio = 0.0
        if normalized_hint and normalized_name:
            ratio = max(ratio, SequenceMatcher(None, normalized_hint, normalized_name).ratio())
        if ascii_hint and ascii_name:
            ratio = max(ratio, SequenceMatcher(None, ascii_hint, ascii_name).ratio())
        if ratio >= 0.70:
            scored.append((ratio, p))

    if not scored:
        return None

    scored.sort(key=lambda item: (item[0], len(item[1].name)), reverse=True)
    return scored[0][1]


def _validate_pdf_path(pdf_path: str) -> str:
    matched = _match_pdf_from_name(pdf_path)
    if matched is None:
        # Retry once after fresh scan in case a new file was uploaded just now.
        matched = _match_pdf_from_name(pdf_path)
    if matched is None:
        raise FileNotFoundError(f"找不到 PDF 檔案：{pdf_path}")
    return str(matched)


def _resolve_active_pdf(pdf_path: str | None, tool_context: ToolContext | None = None) -> str:
    if pdf_path and str(pdf_path).strip():
        requested = str(pdf_path).strip()
        try:
            resolved = _validate_pdf_path(requested)
            if tool_context is not None and hasattr(tool_context, "state"):
                tool_context.state["active_pdf_path"] = resolved
            return resolved
        except FileNotFoundError:
            # ADK may pass placeholders like "This document.pdf" for attachments.
            # In that case, continue to context/artifact inference instead of failing fast.
            if not _looks_like_placeholder_pdf_name(requested):
                direct = _match_pdf_from_name(requested)
                if direct:
                    resolved = str(direct.resolve())
                    if tool_context is not None and hasattr(tool_context, "state"):
                        tool_context.state["active_pdf_path"] = resolved
                    return resolved

    if tool_context is not None and hasattr(tool_context, "state"):
        active = tool_context.state.get("active_pdf_path")
        if active:
            return _validate_pdf_path(str(active))

    # Try infer from message/attachment hints in current context.
    for hint in _extract_pdf_hints_from_context(tool_context):
        matched = _match_pdf_from_name(hint)
        if matched:
            resolved = str(matched.resolve())
            if tool_context is not None and hasattr(tool_context, "state"):
                tool_context.state["active_pdf_path"] = resolved
            return resolved

    pdfs = _discover_pdf_files()
    if len(pdfs) == 1:
        resolved = str(pdfs[0].resolve())
        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["active_pdf_path"] = resolved
        return resolved

    # Prefer freshly uploaded artifact PDFs for ADK web sessions.
    artifact_pdfs = _discover_artifact_pdfs()
    if len(artifact_pdfs) == 1:
        resolved = str(artifact_pdfs[0].resolve())
        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["active_pdf_path"] = resolved
        return resolved

    if len(pdfs) > 1:
        latest = sorted(pdfs, key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        names = ", ".join(p.name for p in latest)
        raise ValueError(
            "目前有多份 PDF，尚未設定 active PDF。"
            "請先用 set_active_pdf(pdf_path) 指定檔案。"
            f"可用檔案（最近）：{names}"
        )

    raise ValueError("尚未指定 PDF 路徑，且找不到可用 PDF。請先上傳檔案或提供 pdf_path。")


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

    lines_out = [f"這份文件共約 {page_count or len(sample_docs)} 頁。"]
    if top_speakers:
        lines_out.append(f"- 主要疑似對話者：{', '.join(top_speakers)}")
    else:
        lines_out.append("- 文件型態：偏敘述/條文型內容，非明確對話格式。")
    if top_keywords:
        lines_out.append(f"- 高頻英文關鍵詞：{', '.join(top_keywords)}")
    if lines:
        preview_line = lines[0]
        if len(preview_line) > 180:
            preview_line = preview_line[:180] + "..."
        lines_out.append(f"- 開頭內容預覽：{preview_line}")
    return "\n".join(lines_out)


def summarize_pdf(pdf_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Summarize what a PDF is about. Good for 'what is this document about' style questions."""
    resolved = _resolve_active_pdf(pdf_path, tool_context)
    runtime = _load_pdf_runtime(resolved)
    if runtime["detected_mode"] == "index":
        answer = answer_with_document_index("請總結這份文件的大意", runtime["doc_index"])
        if answer:
            return answer
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
    latest = _latest_pdf_file()
    if latest:
        resolved = str(latest.resolve())
        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["active_pdf_path"] = resolved
        return f"目前尚未手動設定，已自動選用最新 PDF：{resolved}"
    return "目前尚未設定 active PDF，且找不到可用 PDF。"


def list_available_pdfs(limit: int = 20) -> str:
    """List locally discoverable PDF files for quick selection/debugging."""
    pdfs = sorted(_discover_pdf_files(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs:
        return "目前找不到任何 PDF 檔案。"
    max_items = max(1, min(limit, 50))
    lines = ["目前可用 PDF（依最後修改時間排序）："]
    for p in pdfs[:max_items]:
        lines.append(f"- {p.name} | {p.resolve()}")
    return "\n".join(lines)


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
            return get_pdf_page(pages[0], pdf_path=resolved, tool_context=tool_context)

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

    return search_pdf(question, pdf_path=resolved, top_k=4, tool_context=tool_context)
