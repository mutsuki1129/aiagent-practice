from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from rag.csv_loader import load_csv_and_split
from rag.pdf_loader import load_and_split
from rag.query_engine import (
    answer_with_document_index,
    build_document_index,
    detect_document_mode,
    extract_page_numbers,
    find_heading_matches,
    hybrid_retrieve,
    is_page_query,
)
from rag.vector_store import create_vector_store, get_retriever

try:
    from google.adk.agents.context import Context as ToolContext
except Exception:  # pragma: no cover
    ToolContext = Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADK_ROOT = PROJECT_ROOT / "adk_apps"
ADK_AGENT_ROOT = ADK_ROOT / "pdf_qa_agent"
SUPPORTED_EXTS = {".pdf", ".csv"}
CSV_HIDDEN_FIELDS = {"source_pdf", "note", "page", "pdf", "pdf_path", "pdf_source"}


def _detect_source_type(doc: Any) -> str:
    source_type = str(doc.metadata.get("source_type", "") or "").strip().lower()
    if source_type:
        return source_type
    source = str(doc.metadata.get("source", "") or "").strip().lower()
    if source.endswith(".csv"):
        return "csv"
    return "pdf"


def _format_position_label(doc: Any) -> str:
    pos = int(doc.metadata.get("page", 0) or 0)
    if _detect_source_type(doc) == "csv":
        return f"第 {pos} 列"
    return f"第 {pos} 頁"


def _clean_ocr_text(text: str) -> str:
    cleaned = (text or "").replace("\u3000", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。；：！？、）】」』])", "", cleaned)
    cleaned = re.sub(r"(?<=[（【「『])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    return cleaned


def _extract_csv_fields(doc: Any) -> dict[str, str]:
    fields: dict[str, str] = {}
    content = str(getattr(doc, "page_content", "") or "")
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_text = key.strip()
        if key_text:
            fields[key_text] = value.strip()
    return fields


def _filtered_csv_fields(doc: Any) -> dict[str, str]:
    fields = _extract_csv_fields(doc)
    filtered: dict[str, str] = {}
    for key, value in fields.items():
        if key.strip().lower() in CSV_HIDDEN_FIELDS:
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
    else:
        head = f"你指定的位置是：{'、'.join(labels)}。"
    return f"{head}\n位置完整內容：\n" + "\n\n".join(blocks)


def _format_preview_docs(docs: list[Any], limit: int = 4) -> str:
    if not docs:
        return "找不到相關段落。"
    blocks: list[str] = []
    for doc in docs[:limit]:
        label = _format_position_label(doc)
        if _detect_source_type(doc) == "csv":
            text = _format_csv_doc_content(doc)
        else:
            text = (doc.page_content or "").strip().replace("\n", " ")
        if len(text) > 420:
            text = text[:420] + "..."
        blocks.append(f"[{label}]\n{text}")
    return "\n\n".join(blocks)


@lru_cache(maxsize=16)
def _load_document_runtime(document_path: str) -> dict[str, Any]:
    ext = Path(document_path).suffix.lower()
    if ext == ".pdf":
        docs = load_and_split(document_path)
        doc_index = build_document_index(docs)
        detected_mode = detect_document_mode(doc_index, docs)
    elif ext == ".csv":
        docs = load_csv_and_split(document_path)
        doc_index = {}
        detected_mode = "csv"
    else:
        raise ValueError(f"不支援的檔案格式：{ext}")

    retriever = None
    vector_error = ""
    try:
        retriever = get_retriever(create_vector_store(docs), top_k=4)
    except Exception as exc:  # pragma: no cover
        vector_error = str(exc)
    return {
        "docs": docs,
        "doc_index": doc_index,
        "detected_mode": detected_mode,
        "retriever": retriever,
        "vector_error": vector_error,
        "source_type": "csv" if ext == ".csv" else "pdf",
        "path": document_path,
    }


def _normalize_filename(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"\.(pdf|csv)$", "", normalized)
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)
    return normalized


def _normalize_ascii_filename(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"\.(pdf|csv)$", "", normalized)
    normalized = re.sub(r"[^0-9a-z]+", "", normalized)
    return normalized


def _looks_like_placeholder_name(value: str) -> bool:
    norm = _normalize_ascii_filename(value)
    placeholders = {
        "thisdocument",
        "document",
        "uploadedfile",
        "attachment",
        "file",
        "pdf",
        "csv",
    }
    return norm in placeholders


def _search_roots() -> list[Path]:
    return [
        Path.cwd(),
        PROJECT_ROOT,
        ADK_ROOT,
        ADK_ROOT / ".adk" / "artifacts",
        ADK_AGENT_ROOT / ".adk" / "artifacts",
    ]


def _discover_documents() -> list[Path]:
    files: dict[str, Path] = {}
    skip_dirs = {".git", "venv", "__pycache__", "chroma_db"}

    for root in _search_roots():
        if not root.exists():
            continue

        for ext in SUPPORTED_EXTS:
            for p in root.glob(f"*{ext}"):
                if p.is_file():
                    files[str(p.resolve())] = p.resolve()

        if "artifacts" in root.parts:
            for ext in SUPPORTED_EXTS:
                for p in root.rglob(f"*{ext}"):
                    if p.is_file():
                        files[str(p.resolve())] = p.resolve()
            continue

        for child in root.iterdir():
            if not child.is_dir() or child.name in skip_dirs:
                continue
            for ext in SUPPORTED_EXTS:
                for p in child.glob(f"*{ext}"):
                    if p.is_file():
                        files[str(p.resolve())] = p.resolve()

    return sorted(files.values(), key=lambda p: str(p))


def _discover_files_by_ext(ext: str) -> list[Path]:
    ext = ext.lower()
    return [p for p in _discover_documents() if p.suffix.lower() == ext]


def _latest_document_file() -> Path | None:
    docs = _discover_documents()
    if not docs:
        return None
    return max(docs, key=lambda p: p.stat().st_mtime)


def _extract_hints_from_context(tool_context: ToolContext | None) -> list[str]:
    if tool_context is None:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _push(text: str) -> None:
        for match in re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff ()]+\.(?:pdf|csv)", text, flags=re.I):
            key = match.strip()
            if key and key not in seen:
                seen.add(key)
                candidates.append(key)

    for attr in ("state", "session", "request", "invocation", "user_content", "new_message"):
        value = getattr(tool_context, attr, None)
        if value is not None:
            _push(str(value))

    _push(str(tool_context))
    return candidates


def _match_document_from_name(file_hint: str, *, prefer_ext: str | None = None) -> Path | None:
    hint = (file_hint or "").strip().strip('"').strip("'")
    if not hint:
        return None

    hint_path = Path(hint).expanduser()
    candidate_paths: list[Path] = []
    if hint_path.is_absolute():
        candidate_paths.append(hint_path)
    else:
        candidate_paths.extend([
            Path.cwd() / hint_path,
            PROJECT_ROOT / hint_path,
            ADK_ROOT / hint_path,
        ])

    for candidate in candidate_paths:
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate.suffix.lower() in SUPPORTED_EXTS:
            if prefer_ext is None or candidate.suffix.lower() == prefer_ext:
                return candidate.resolve()

    files = _discover_documents()
    if prefer_ext:
        files = [p for p in files if p.suffix.lower() == prefer_ext]
    if not files:
        return None

    basename = Path(hint).name
    for p in files:
        if p.name == basename:
            return p

    normalized_hint = _normalize_filename(basename or hint)
    ascii_hint = _normalize_ascii_filename(basename or hint)
    if not normalized_hint and not ascii_hint:
        return None

    scored: list[tuple[float, Path]] = []
    for p in files:
        normalized_name = _normalize_filename(p.name)
        ascii_name = _normalize_ascii_filename(p.name)
        if normalized_hint and normalized_name and normalized_hint == normalized_name:
            return p
        if ascii_hint and ascii_name and ascii_hint == ascii_name:
            return p

        ratio = 0.0
        if normalized_hint and normalized_name:
            if normalized_hint in normalized_name or normalized_name in normalized_hint:
                ratio = max(ratio, 0.95)
            ratio = max(ratio, SequenceMatcher(None, normalized_hint, normalized_name).ratio())
        if ascii_hint and ascii_name:
            if ascii_hint in ascii_name or ascii_name in ascii_hint:
                ratio = max(ratio, 0.94)
            ratio = max(ratio, SequenceMatcher(None, ascii_hint, ascii_name).ratio())
        if ratio >= 0.70:
            scored.append((ratio, p))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], len(item[1].name)), reverse=True)
    return scored[0][1]


def _validate_document_path(file_path: str, *, prefer_ext: str | None = None) -> str:
    matched = _match_document_from_name(file_path, prefer_ext=prefer_ext)
    if matched is None:
        matched = _match_document_from_name(file_path, prefer_ext=prefer_ext)
    if matched is None:
        ext_hint = prefer_ext or "pdf/csv"
        raise FileNotFoundError(f"找不到 {ext_hint} 檔案：{file_path}")
    return str(matched)


def _resolve_active_document(
    file_path: str | None,
    tool_context: ToolContext | None = None,
    *,
    prefer_ext: str | None = None,
) -> str:
    if file_path and str(file_path).strip():
        requested = str(file_path).strip()
        try:
            resolved = _validate_document_path(requested, prefer_ext=prefer_ext)
            if tool_context is not None and hasattr(tool_context, "state"):
                tool_context.state["active_document_path"] = resolved
                tool_context.state["active_pdf_path"] = resolved  # backward compatibility
            return resolved
        except FileNotFoundError:
            if not _looks_like_placeholder_name(requested):
                direct = _match_document_from_name(requested, prefer_ext=prefer_ext)
                if direct:
                    resolved = str(direct.resolve())
                    if tool_context is not None and hasattr(tool_context, "state"):
                        tool_context.state["active_document_path"] = resolved
                        tool_context.state["active_pdf_path"] = resolved
                    return resolved

    if tool_context is not None and hasattr(tool_context, "state"):
        active = tool_context.state.get("active_document_path") or tool_context.state.get("active_pdf_path")
        if active:
            return _validate_document_path(str(active), prefer_ext=prefer_ext)

    for hint in _extract_hints_from_context(tool_context):
        matched = _match_document_from_name(hint, prefer_ext=prefer_ext)
        if matched:
            resolved = str(matched.resolve())
            if tool_context is not None and hasattr(tool_context, "state"):
                tool_context.state["active_document_path"] = resolved
                tool_context.state["active_pdf_path"] = resolved
            return resolved

    files = _discover_documents()
    if prefer_ext:
        files = [p for p in files if p.suffix.lower() == prefer_ext]

    if len(files) == 1:
        resolved = str(files[0].resolve())
        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["active_document_path"] = resolved
            tool_context.state["active_pdf_path"] = resolved
        return resolved

    if len(files) > 1:
        latest = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        names = ", ".join(p.name for p in latest)
        raise ValueError(
            "目前有多份文件，尚未設定 active 文件。"
            "請先用 set_active_document(file_path) 指定檔案。"
            f"可用檔案（最近）：{names}"
        )

    raise ValueError("尚未指定檔案路徑，且找不到可用檔案。請先上傳檔案或提供 file_path。")


def _build_general_summary(docs: list[Any], *, source_type: str) -> str:
    unit = "列" if source_type == "csv" else "頁"
    count = len({int(doc.metadata.get("page", 0) or 0) for doc in docs if doc.metadata.get("page")})
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

    word_counts: Counter[str] = Counter()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]{3,}", text.lower()):
        word_counts[token] += 1

    top_keywords = [word for word, _ in word_counts.most_common(6)]
    top_speakers = [name for name, _ in speaker_counts.most_common(4)]

    lines_out = [f"這份文件共約 {count or len(sample_docs)} {unit}。"]
    if top_speakers:
        lines_out.append(f"- 主要疑似對話者：{', '.join(top_speakers)}")
    if top_keywords:
        lines_out.append(f"- 高頻關鍵詞：{', '.join(top_keywords)}")
    if lines:
        preview_line = lines[0]
        if len(preview_line) > 180:
            preview_line = preview_line[:180] + "..."
        lines_out.append(f"- 開頭內容預覽：{preview_line}")
    return "\n".join(lines_out)


def _extract_row_numbers(query: str) -> list[int]:
    text = query.strip().lower()
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


def _is_row_query(query: str) -> bool:
    return bool(_extract_row_numbers(query))


def _try_csv_direct_answer(question: str, docs: list[Any]) -> str | None:
    csv_docs = [doc for doc in docs if _detect_source_type(doc) == "csv"]
    if not csv_docs:
        return None

    code_match = re.search(r"\b[0-9A-Z]{4,6}\b", question.upper())
    code = code_match.group(0) if code_match else ""

    if code:
        for doc in csv_docs:
            fields = _filtered_csv_fields(doc)
            if fields.get("code", "").strip().upper() == code:
                title = _clean_ocr_text(fields.get("title", "") or fields.get("title_excerpt", ""))
                definition = _clean_ocr_text(fields.get("definition", ""))
                lines = ["已在 CSV 中找到相符資料。", f"- 代碼：{code}"]
                if title:
                    lines.append(f"- 標題：{title}")
                if definition:
                    lines.append(f"- 定義：{definition}")
                lines.append(f"- 來源：{_format_position_label(doc)}")
                return "\n".join(lines)

    normalized_q = _clean_ocr_text(question).lower()
    terms = [t for t in re.findall(r"[0-9a-z\u4e00-\u9fff]{2,}", normalized_q) if t]
    scored: list[tuple[int, Any]] = []
    for doc in csv_docs:
        text = _clean_ocr_text(str(getattr(doc, "page_content", "") or "")).lower()
        score = 0
        for term in terms:
            if term in text:
                score += 2
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    picked = [item[1] for item in scored[:2]]
    if not picked:
        return None
    return "已在 CSV 中找到可能相關資料：\n" + _format_preview_docs(picked, limit=2)


def _keyword_search_docs(query: str, docs: list[Any], top_k: int = 4) -> list[Any]:
    normalized = _clean_ocr_text(query).lower()
    terms = [t for t in re.findall(r"[0-9a-z\u4e00-\u9fff]{2,}", normalized) if t]
    if not terms:
        return docs[:top_k]
    scored: list[tuple[int, Any]] = []
    for doc in docs:
        text = _clean_ocr_text(str(getattr(doc, "page_content", "") or "")).lower()
        score = 0
        for term in terms:
            if term in text:
                score += 2
        if normalized and normalized in text:
            score += 3
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]] if scored else docs[:top_k]


# -------------------------
# New document tools (PDF/CSV)
# -------------------------

def set_active_document(file_path: str, tool_context: ToolContext | None = None) -> str:
    """Set the active document (PDF/CSV) for this ADK session."""
    resolved = _resolve_active_document(file_path, tool_context)
    return f"已設定目前文件：{resolved}"


def get_active_document(tool_context: ToolContext | None = None) -> str:
    """Return the current active document path."""
    if tool_context is not None and hasattr(tool_context, "state"):
        active = tool_context.state.get("active_document_path") or tool_context.state.get("active_pdf_path")
        if active:
            return f"目前使用中的文件：{active}"
    latest = _latest_document_file()
    if latest:
        resolved = str(latest.resolve())
        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["active_document_path"] = resolved
            tool_context.state["active_pdf_path"] = resolved
        return f"目前尚未手動設定，已自動選用最新文件：{resolved}"
    return "目前尚未設定 active 文件，且找不到可用檔案。"


def list_available_documents(limit: int = 20) -> str:
    """List locally discoverable PDF/CSV files for quick selection."""
    files = sorted(_discover_documents(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return "目前找不到任何 PDF/CSV 檔案。"
    max_items = max(1, min(limit, 50))
    lines = ["目前可用文件（依最後修改時間排序）："]
    for p in files[:max_items]:
        lines.append(f"- {p.name} | {p.resolve()}")
    return "\n".join(lines)


def summarize_document(file_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Summarize the active document (PDF/CSV)."""
    resolved = _resolve_active_document(file_path, tool_context)
    runtime = _load_document_runtime(resolved)
    if runtime["source_type"] == "pdf" and runtime["detected_mode"] == "index":
        answer = answer_with_document_index("請總結這份文件的大意", runtime["doc_index"])
        if answer:
            return answer
    return _build_general_summary(runtime["docs"], source_type=runtime["source_type"])


def get_document_position(position_number: int, file_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Read full content of a specific page (PDF) or row (CSV)."""
    resolved = _resolve_active_document(file_path, tool_context)
    runtime = _load_document_runtime(resolved)
    docs = [doc for doc in runtime["docs"] if int(doc.metadata.get("page", 0) or 0) == int(position_number)]
    if not docs:
        unit = "列" if runtime["source_type"] == "csv" else "頁"
        return f"文件中找不到第 {position_number} {unit}。"
    return _summarize_position_docs(docs, position_kind="row" if runtime["source_type"] == "csv" else "page")


def search_document(query: str, file_path: str | None = None, top_k: int = 4, tool_context: ToolContext | None = None) -> str:
    """Search active PDF/CSV and return relevant passages with page/row labels."""
    resolved = _resolve_active_document(file_path, tool_context)
    runtime = _load_document_runtime(resolved)
    max_k = max(1, min(top_k, 8))
    if runtime["retriever"] is not None:
        scored = hybrid_retrieve(query.strip(), runtime["docs"], runtime["retriever"], top_k=max_k)
        docs = [item.document if hasattr(item, "document") else item for item in scored]
    else:
        docs = _keyword_search_docs(query, runtime["docs"], top_k=max_k)
    return _format_preview_docs(docs, limit=max(1, min(top_k, 8)))


def lookup_document_index(question: str, file_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Use structured index lookup for index-like PDF. CSV is not index mode."""
    resolved = _resolve_active_document(file_path, tool_context)
    runtime = _load_document_runtime(resolved)
    if runtime["source_type"] == "csv":
        return "目前文件是 CSV，請改用 search_document 或 answer_document_question。"
    if runtime["detected_mode"] != "index":
        return "這份文件不是索引型 PDF，請改用 search_document 或 summarize_document。"
    answer = answer_with_document_index(question.strip(), runtime["doc_index"])
    return answer or "索引中找不到明確答案。"


def answer_document_question(question: str, file_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Answer a question using evidence from active PDF/CSV."""
    resolved = _resolve_active_document(file_path, tool_context)
    runtime = _load_document_runtime(resolved)
    question = question.strip()

    if runtime["source_type"] == "csv" and _is_row_query(question):
        rows = _extract_row_numbers(question)
        if rows:
            return get_document_position(rows[0], file_path=resolved, tool_context=tool_context)

    if runtime["source_type"] == "pdf" and is_page_query(question):
        pages = extract_page_numbers(question)
        if pages:
            return get_document_position(pages[0], file_path=resolved, tool_context=tool_context)

    if runtime["source_type"] == "csv":
        direct = _try_csv_direct_answer(question, runtime["docs"])
        if direct:
            return direct
        return search_document(question, file_path=resolved, top_k=4, tool_context=tool_context)

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
            return _format_preview_docs(docs, limit=3)

    return search_document(question, file_path=resolved, top_k=4, tool_context=tool_context)


# -------------------------
# Backward-compatible PDF tools
# -------------------------

def set_active_pdf(pdf_path: str, tool_context: ToolContext | None = None) -> str:
    """Backward-compatible alias: set active file (PDF/CSV accepted)."""
    return set_active_document(pdf_path, tool_context=tool_context)


def get_active_pdf(tool_context: ToolContext | None = None) -> str:
    """Backward-compatible alias for get_active_document."""
    return get_active_document(tool_context=tool_context)


def list_available_pdfs(limit: int = 20) -> str:
    """Backward-compatible alias. Now lists PDF and CSV files."""
    return list_available_documents(limit=limit)


def summarize_pdf(pdf_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Backward-compatible alias for summarize_document."""
    return summarize_document(file_path=pdf_path, tool_context=tool_context)


def get_pdf_page(page_number: int, pdf_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Backward-compatible alias for get_document_position."""
    return get_document_position(position_number=page_number, file_path=pdf_path, tool_context=tool_context)


def lookup_pdf_index(question: str, pdf_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Backward-compatible alias for lookup_document_index."""
    return lookup_document_index(question=question, file_path=pdf_path, tool_context=tool_context)


def search_pdf(query: str, pdf_path: str | None = None, top_k: int = 4, tool_context: ToolContext | None = None) -> str:
    """Backward-compatible alias for search_document."""
    return search_document(query=query, file_path=pdf_path, top_k=top_k, tool_context=tool_context)


def answer_pdf_question(question: str, pdf_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Backward-compatible alias for answer_document_question."""
    return answer_document_question(question=question, file_path=pdf_path, tool_context=tool_context)


# -------------------------
# Optional CSV convenience tools
# -------------------------

def set_active_csv(csv_path: str, tool_context: ToolContext | None = None) -> str:
    """Set active CSV file explicitly."""
    resolved = _resolve_active_document(csv_path, tool_context, prefer_ext=".csv")
    return f"已設定目前 CSV：{resolved}"


def get_csv_row(row_number: int, csv_path: str | None = None, tool_context: ToolContext | None = None) -> str:
    """Read full content of a specific CSV row."""
    resolved = _resolve_active_document(csv_path, tool_context, prefer_ext=".csv")
    return get_document_position(position_number=row_number, file_path=resolved, tool_context=tool_context)


def list_available_csvs(limit: int = 20) -> str:
    """List discoverable CSV files only."""
    files = sorted(_discover_files_by_ext('.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return "目前找不到任何 CSV 檔案。"
    max_items = max(1, min(limit, 50))
    lines = ["目前可用 CSV（依最後修改時間排序）："]
    for p in files[:max_items]:
        lines.append(f"- {p.name} | {p.resolve()}")
    return "\n".join(lines)
