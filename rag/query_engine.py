from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from langchain_core.documents import Document


SUMMARY_MARKERS = (
    "大綱",
    "摘要",
    "總結",
    "重點",
    "整理",
    "這份文件在講什麼",
    "這份文件主要在講什麼",
    "overview",
    "outline",
    "summary",
    "summarize",
)

LOOKUP_NOISE_PATTERNS = (
    "請問",
    "可以告訴我",
    "請幫我",
    "這份文件",
    "主要在講什麼",
    "在講什麼",
    "是什麼",
    "是甚麼",
    "有哪些",
    "有哪一些",
    "屬於哪一類",
    "在哪一類",
    "哪一類",
    "分類是什麼",
    "的代碼",
    "代碼是什麼",
    "代碼是甚麼",
    "相關分類",
    "的分類",
)

BAD_REPRESENTATIVE_KEYWORDS = (
    "特徵",
    "症狀表現",
    "還包括",
    "可用於",
    "診斷標準",
    "表現",
    "包括",
    "分類可",
)

MAIN_SECTION_TITLE_KEYWORDS = (
    "障礙",
    "症",
    "緊張症",
)


@dataclass(slots=True)
class ScoredDocument:
    document: Document
    score: float
    reasons: list[str]


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _normalize_lookup_phrase(text: str) -> str:
    normalized = normalize_text(text)
    replacements = (
        ("不適應", "不適"),
        ("身體體驗", "身體體驗"),
        ("有哪些", ""),
        ("有哪一些", ""),
        ("相關分類", ""),
        ("分類", ""),
        ("哪一類", ""),
        ("是什麼", ""),
        ("是甚麼", ""),
        ("?", ""),
        ("？", ""),
    )
    for src, dst in replacements:
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _clean_section_title(title: str) -> str:
    cleaned = re.sub(r"…+", "", title).strip()
    cleaned = re.sub(r"^\d+\.", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_inline_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_section_title_useful(title: str) -> bool:
    if not title:
        return False
    if len(title) < 4 or len(title) > 40:
        return False
    if any(mark in title for mark in ("。", "，", "：", ":", "（", "）")):
        return False
    if re.search(r"\d{2,}", title):
        return False
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", title)
    if len(chinese_chars) < 3:
        return False
    bad_fragments = ("百分位", "標準化測試", "臨床判斷", "適當可", "根據", "如果沒有")
    if any(fragment in title for fragment in bad_fragments):
        return False
    return True


def _is_main_section_title(title: str) -> bool:
    if not _is_section_title_useful(title):
        return False
    if title.startswith("第") and "章" in title:
        return False
    if title.startswith("附錄") or title.startswith("索引"):
        return False
    return any(keyword in title for keyword in MAIN_SECTION_TITLE_KEYWORDS)


def _extract_heading_summary_from_content(code: str, title: str, content: str) -> str:
    start_match = re.search(
        rf"{re.escape(code)}\s+{re.escape(title)}[：:]\s*",
        content,
        flags=re.S,
    )
    if not start_match:
        return ""

    remainder = content[start_match.end():]
    terminator_patterns = (
        r"\n\s*[0-9A-Z]{4,6}\s+[^\n：:]{2,80}[：:]",
        r"\n\s*[^\n：:]{2,40}…{3,}\s*(?:[0-9A-Z]{4,6})?\s*$",
        r"\n\s*\d+\.\s*[^\n]{2,40}$",
        r"\n\s*[一二三四五六七八九十]+\s*[、.．]\s*[^\n]{2,40}$",
    )

    end_positions = [len(remainder)]
    for pattern in terminator_patterns:
        match = re.search(pattern, remainder, flags=re.M)
        if match:
            end_positions.append(match.start())

    summary = remainder[: min(end_positions)]
    return _normalize_inline_whitespace(summary)


def _is_representative_heading(item: dict[str, Any]) -> bool:
    title = str(item.get("title", "")).strip()
    if not title:
        return False
    if len(title) < 2 or len(title) > 32:
        return False
    if any(keyword in title for keyword in BAD_REPRESENTATIVE_KEYWORDS):
        return False
    if any(mark in title for mark in ("：", ":", "（", "）", "。")):
        return False
    if title.endswith("診斷") or title.endswith("表現"):
        return False
    return True


def _select_summary_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for item in sections:
        title = item["title"]
        if title in seen_titles:
            continue
        if not _is_main_section_title(title):
            continue
        seen_titles.add(title)
        filtered.append(item)
    return filtered[:10]


def build_document_index(all_docs: list[Document]) -> dict[str, Any]:
    headings: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    by_page: dict[int, list[Document]] = {}
    current_section = ""

    for doc in all_docs:
        page = int(doc.metadata.get("page", 0) or 0)
        by_page.setdefault(page, []).append(doc)
        text = doc.page_content or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        leading_section = ""
        trailing_section = ""

        for line in lines[:8]:
            section_only_match = re.match(r"^(.{4,80}?)…+$", line)
            if section_only_match:
                candidate = _clean_section_title(section_only_match.group(1))
                if _is_section_title_useful(candidate):
                    leading_section = candidate
                    sections.append({"title": candidate, "start_code": "", "page": page})
                    break

            subsection_match = re.match(r"^\d+\.\s*(.{2,80})$", line)
            if subsection_match:
                candidate = _clean_section_title(subsection_match.group(1))
                if _is_section_title_useful(candidate):
                    leading_section = candidate
                    sections.append({"title": candidate, "start_code": "", "page": page})
                    break

        if leading_section:
            current_section = leading_section

        for match in re.finditer(r"\b([0-9A-Z]{4,6})\s+([^：:\n]{2,80})[：:]", text):
            code = match.group(1).strip()
            title = _normalize_inline_whitespace(match.group(2))
            headings.append(
                {
                    "code": code,
                    "title": title,
                    "page": page,
                    "source": doc.metadata.get("source", "未知檔案"),
                    "section": current_section,
                    "summary": "",
                }
            )

        for section_match in re.finditer(r"([^\n：:]{4,80}?)…+\s*([0-9A-Z]{4,6})", text):
            candidate = _clean_section_title(section_match.group(1))
            if _is_section_title_useful(candidate):
                sections.append(
                    {
                        "title": candidate,
                        "start_code": section_match.group(2).strip(),
                        "page": page,
                    }
                )

        for line in lines[-6:]:
            section_only_match = re.match(r"^(.{4,80}?)…+$", line)
            if section_only_match:
                candidate = _clean_section_title(section_only_match.group(1))
                if _is_section_title_useful(candidate):
                    trailing_section = candidate

        if trailing_section:
            current_section = trailing_section

    for heading in headings:
        page_docs = by_page.get(int(heading["page"]), [])
        summary = ""
        for doc in page_docs:
            summary = _extract_heading_summary_from_content(
                heading["code"],
                heading["title"],
                doc.page_content or "",
            )
            if summary:
                break
        heading["summary"] = summary

    seen_sections: set[tuple[str, int]] = set()
    deduped_sections: list[dict[str, Any]] = []
    for item in sections:
        title = item["title"]
        if not title:
            continue
        key = (title, item["page"])
        if key in seen_sections:
            continue
        seen_sections.add(key)
        deduped_sections.append(item)

    for section in deduped_sections:
        if section["start_code"]:
            continue
        for heading in headings:
            if heading["page"] == section["page"] and heading["section"] == section["title"]:
                section["start_code"] = heading["code"]
                break

    return {
        "headings": headings,
        "sections": deduped_sections,
        "summary_sections": _select_summary_sections(deduped_sections),
        "page_count": len(by_page),
    }


def is_summary_query(query: str) -> bool:
    lowered = normalize_text(query)
    return any(marker in lowered for marker in SUMMARY_MARKERS)


def extract_page_numbers(query: str) -> list[int]:
    matches = re.findall(r"第\s*(\d{1,3})\s*頁", query)
    pages = [int(item) for item in matches]
    deduped: list[int] = []
    for page in pages:
        if page not in deduped:
            deduped.append(page)
    return deduped


def is_page_query(query: str) -> bool:
    return "第" in query and "頁" in query and bool(extract_page_numbers(query))


def should_use_literal_lookup(query: str) -> bool:
    normalized = normalize_text(query)
    if not normalized or is_summary_query(query):
        return False
    if re.search(r"\b[0-9A-Z]{4,6}\b", query.upper()):
        return True
    token_count = len([token for token in normalized.split(" ") if token])
    return len(normalized) <= 24 and token_count <= 4


def build_overview_context(doc_index: dict[str, Any]) -> str:
    sections = doc_index.get("summary_sections") or doc_index.get("sections", [])
    if not sections:
        return "文件中未擷取到結構化分類標題。"

    lines: list[str] = []
    for item in sections[:20]:
        start_code = item["start_code"] or "?"
        lines.append(f"- {item['title']}（起始代碼 {start_code}，第 {item['page']} 頁）")
    return "\n".join(lines)


def _collect_section_items(section_title: str, headings: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    matched = [item for item in headings if item.get("section") == section_title and _is_representative_heading(item)]
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in matched:
        key = f"{item['code']}::{item['title']}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _build_section_highlight(section_title: str, headings: list[dict[str, Any]]) -> str:
    items = _collect_section_items(section_title, headings, limit=3)
    if not items:
        return "本章節在文件中有分類條目，但目前未擷取到足夠乾淨的代表性子項。"

    names = [item["title"] for item in items if item.get("title")]
    return "代表性條目包含：" + "、".join(names) + "。"


def build_llm_prompt(query: str, context_docs: list[Document], doc_index: dict[str, Any]) -> str:
    context_blocks: list[str] = []
    for doc in context_docs[:8]:
        page = doc.metadata.get("page", "?")
        text = (doc.page_content or "").strip().replace("\n", " ")
        context_blocks.append(f"[第 {page} 頁]\n{text}")

    overview = build_overview_context(doc_index)
    joined_context = "\n\n".join(context_blocks)
    return (
        "你是 ICD-11 文件問答助手，請只根據提供的文件內容回答。\n"
        "回答規則：\n"
        "1. 使用繁體中文。\n"
        "2. 若沒有充分證據，明確說找不到。\n"
        "3. 若有可能，補上代碼、分類與頁碼。\n\n"
        f"文件分類概覽：\n{overview}\n\n"
        f"檢索段落：\n{joined_context}\n\n"
        f"使用者問題：{query.strip()}"
    )


def build_summary_prompt(query: str, doc_index: dict[str, Any], all_docs: list[Document]) -> str:
    overview = build_overview_context(doc_index)
    intro_blocks: list[str] = []
    for doc in all_docs[:4]:
        page = doc.metadata.get("page", "?")
        text = (doc.page_content or "").strip().replace("\n", " ")
        intro_blocks.append(f"[第 {page} 頁] {text[:360]}")

    context = "\n\n".join(intro_blocks)
    return (
        "你是 ICD-11 文件分析助手，請根據文件內容回答。\n"
        "請先用 4-8 點條列整理重點，再補 1 段簡短總結，使用繁體中文。\n\n"
        f"文件分類概覽：\n{overview}\n\n"
        f"文件前段內容：\n{context}\n\n"
        f"使用者問題：{query.strip()}"
    )


def extract_lookup_terms(query: str) -> list[str]:
    cleaned = query
    for noise in LOOKUP_NOISE_PATTERNS:
        cleaned = cleaned.replace(noise, " ")
    cleaned = re.sub(r"[？?.,，、：:()（）「」/]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    terms: list[str] = []
    codes = re.findall(r"\b[0-9A-Z]{4,6}\b", cleaned.upper())
    terms.extend(codes)

    chinese_runs = re.findall(r"[\u4e00-\u9fff]{2,}", cleaned)
    latin_runs = re.findall(r"[A-Za-z][A-Za-z0-9+-]{1,}", cleaned)
    for item in chinese_runs + latin_runs:
        if item not in terms:
            terms.append(item)
        if item.endswith("障礙") and item[:-2] and item[:-2] not in terms:
            terms.append(item[:-2])
    if cleaned and cleaned not in terms:
        terms.append(cleaned)
    return terms[:10]


def _shared_character_score(needle: str, haystack: str) -> float:
    ignored = {"障", "礙", "症", "病", "性", "類", "型"}
    needle_chars = {char for char in needle if "\u4e00" <= char <= "\u9fff" and char not in ignored}
    if not needle_chars:
        return 0.0
    haystack_chars = {char for char in haystack if "\u4e00" <= char <= "\u9fff" and char not in ignored}
    overlap = len(needle_chars & haystack_chars)
    return overlap / max(len(needle_chars), 1)


def find_heading_matches(query: str, doc_index: dict[str, Any]) -> list[dict[str, Any]]:
    headings = doc_index.get("headings", [])
    terms = extract_lookup_terms(query)
    matches: list[tuple[float, dict[str, Any]]] = []

    for heading in headings:
        haystack = " ".join(
            [
                heading.get("code", ""),
                heading.get("title", ""),
                heading.get("section", ""),
            ]
        ).lower()
        score = 0.0
        for term in terms:
            probe = term.lower()
            if probe and probe in haystack:
                score += 10.0 if probe == heading.get("code", "").lower() else min(8.0, 2.0 + len(probe) * 0.5)
            else:
                fuzzy = _shared_character_score(probe, haystack)
                if len(probe) >= 4 and fuzzy >= 0.75:
                    score += 4.0
        if score > 0:
            matches.append((score, heading))

    matches.sort(key=lambda item: item[0], reverse=True)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, item in matches:
        key = f"{item['code']}::{item['title']}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def find_section_matches(query: str, doc_index: dict[str, Any]) -> list[dict[str, Any]]:
    sections = doc_index.get("sections", [])
    matches: list[dict[str, Any]] = []
    normalized_query = _normalize_lookup_phrase(query)
    for item in sections:
        title = item.get("title", "")
        if not title:
            continue
        normalized_title = _normalize_lookup_phrase(title)
        if title in query or title.lower() in query.lower():
            matches.append(item)
            continue
        if normalized_query and normalized_title and (
            normalized_query in normalized_title or normalized_title in normalized_query
        ):
            matches.append(item)
            continue
        if "物質使用" in query and "物質使用或成癮行為所致障礙" in title:
            matches.append(item)
            continue
        if _shared_character_score(title, query) >= 0.8:
            matches.append(item)
    return matches


def _format_heading_details(item: dict[str, Any], heading: str | None = None) -> str:
    lines: list[str] = []
    if heading:
        lines.append(heading)
    lines.append(f"- 名稱：`{item['title']}`")
    lines.append(f"- 代碼：`{item['code']}`")
    if item.get("section"):
        lines.append(f"- 所屬分類：`{item['section']}`")
    lines.append(f"- 頁碼：第 {item['page']} 頁")
    if item.get("summary"):
        lines.append(f"- 定義：{item['summary']}")
    else:
        lines.append("- 定義：文件中找到該條目，但未成功擷取完整定義段落。")
    return "\n".join(lines)


def _format_heading_list(title: str, items: list[dict[str, Any]]) -> str:
    lines = [title, f"- 共找到 `{len(items)}` 筆相關條目"]
    for item in items:
        detail_parts = [f"第 {item['page']} 頁"]
        if item.get("section"):
            detail_parts.append(f"屬於 {item['section']}")
        lines.append(f"- `{item['code']}` {item['title']}（{'，'.join(detail_parts)}）")
    return "\n".join(lines)


def _is_list_query(query: str) -> bool:
    return any(marker in query for marker in ("有哪些", "有哪一些", "相關分類"))


def is_code_or_lookup_query(query: str) -> bool:
    lowered = normalize_text(query)
    lookup_markers = (
        "代碼",
        "哪一類",
        "屬於哪一類",
        "分類",
        "相關分類",
        "有哪些",
        "有哪一些",
        "是什麼",
        "是甚麼",
    )
    if re.search(r"\b[0-9A-Z]{4,6}\b", query.upper()):
        return True
    return any(marker in lowered for marker in lookup_markers)


def answer_with_document_index(query: str, doc_index: dict[str, Any]) -> str:
    heading_matches = find_heading_matches(query, doc_index)
    section_matches = find_section_matches(query, doc_index)
    headings = doc_index.get("headings", [])

    if is_summary_query(query) or "主要在講什麼" in query:
        sections = doc_index.get("summary_sections") or doc_index.get("sections", [])
        page_count = doc_index.get("page_count", 0)
        lines = [
            f"這份文件是 ICD-11 精神、行為或神經發育障礙分類索引，共整理 {page_count} 頁內容。",
        ]
        if sections:
            lines.append("主要章節大綱如下：")
            for item in sections[:10]:
                start_code = item["start_code"] or "?"
                lines.append(f"- `{item['title']}`")
                lines.append(f"  頁碼：第 {item['page']} 頁")
                lines.append(f"  起始代碼：`{start_code}`")
                lines.append(f"  重點：{_build_section_highlight(item['title'], headings)}")
            lines.append(
                "\n總結：這份文件以 ICD-11 精神、行為與神經發育障礙的主要診斷索引為主，適合用來快速查找病名、代碼、所屬分類與對應頁碼。"
            )
        return "\n".join(lines)

    if _is_list_query(query):
        if "物質使用所致障礙" in query or "物質使用障礙" in query:
            related = [item for item in headings if "使用所致障礙" in item.get("title", "")]
            if related:
                return _format_heading_list("`物質使用所致障礙` 相關分類如下：", related)

        if section_matches:
            target = section_matches[0]["title"]
            related = [item for item in headings if item.get("section") == target]
            if related:
                return _format_heading_list(f"`{target}` 相關分類如下：", related)

        lookup_terms = extract_lookup_terms(query)
        if lookup_terms:
            related = []
            for item in headings:
                title = item.get("title", "")
                if any(term and term in title for term in lookup_terms):
                    related.append(item)
            if related:
                seen_keys: set[str] = set()
                deduped_related: list[dict[str, Any]] = []
                for item in related:
                    key = f"{item['code']}::{item['title']}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    deduped_related.append(item)
                return _format_heading_list("文件中找到的相關條目如下：", deduped_related)

        if len(heading_matches) > 1:
            dominant_section = heading_matches[0].get("section", "")
            if dominant_section:
                same_section = [item for item in headings if item.get("section") == dominant_section]
                if len(same_section) > 1:
                    return _format_heading_list(f"`{dominant_section}` 相關分類如下：", same_section)
            return _format_heading_list("文件中找到的相關條目如下：", heading_matches)

    if "代碼" in query and heading_matches:
        return _format_heading_details(heading_matches[0], heading="找到對應條目如下：")

    if ("是什麼" in query or "是甚麼" in query) and heading_matches:
        return _format_heading_details(heading_matches[0], heading="找到對應條目如下：")

    if ("哪一類" in query or "屬於哪一類" in query) and heading_matches:
        return _format_heading_details(heading_matches[0], heading="找到對應條目如下：")

    if heading_matches:
        return _format_heading_details(heading_matches[0], heading="文件中找到最接近的條目如下：")

    return ""


def summarize_page_docs(page_docs: list[Document]) -> str:
    if not page_docs:
        return "尚無檢索結果。"

    pages = sorted({int(doc.metadata.get("page", 0) or 0) for doc in page_docs if doc.metadata.get("page")})
    lines = [f"你指定的頁面是：{'、'.join(f'第 {page} 頁' for page in pages)}。"]
    lines.append("頁面完整內容：")
    for doc in page_docs:
        page = doc.metadata.get("page", "?")
        text = (doc.page_content or "").strip()
        lines.append(f"[第 {page} 頁]")
        lines.append(text)
    return "\n".join(lines)


def _token_overlap_score(query_terms: list[str], content: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    for term in query_terms:
        if len(term) < 2:
            continue
        if term in content:
            bonus = 8.0 if re.fullmatch(r"[0-9A-Z]{4,6}", term) else min(6.0, 1.5 + len(term) * 0.6)
            score += bonus
            reasons.append(f"命中字詞:{term}")
    return score, reasons


def hybrid_retrieve(
    query: str,
    all_docs: list[Document],
    retriever,
    top_k: int = 6,
) -> list[ScoredDocument]:
    query_terms = extract_lookup_terms(query)
    ranked: dict[str, ScoredDocument] = {}

    for doc in all_docs:
        content = normalize_text(doc.page_content or "")
        score, reasons = _token_overlap_score(query_terms, content)
        if score <= 0:
            continue
        key = f"{doc.metadata.get('source')}::{doc.metadata.get('page')}::{hash(doc.page_content or '')}"
        ranked[key] = ScoredDocument(document=doc, score=score, reasons=reasons)

    try:
        vector_hits = retriever.invoke(query)
    except Exception:
        vector_hits = []

    for rank, doc in enumerate(vector_hits, start=1):
        key = f"{doc.metadata.get('source')}::{doc.metadata.get('page')}::{hash(doc.page_content or '')}"
        boost = max(0.0, 4.0 - (rank - 1) * 0.6)
        if key in ranked:
            ranked[key].score += boost
            ranked[key].reasons.append(f"向量檢索第{rank}名")
        else:
            ranked[key] = ScoredDocument(document=doc, score=boost, reasons=[f"向量檢索第{rank}名"])

    ordered = sorted(ranked.values(), key=lambda item: item.score, reverse=True)
    return ordered[:top_k]


def detect_document_mode(doc_index: dict[str, Any], all_docs: list[Document]) -> str:
    headings = doc_index.get("headings", [])
    sections = doc_index.get("sections", [])
    summary_sections = doc_index.get("summary_sections", [])
    page_count = int(doc_index.get("page_count", 0) or 0)

    if not all_docs:
        return "general"

    code_heading_count = sum(1 for item in headings if re.fullmatch(r"[0-9A-Z]{4,6}", str(item.get("code", ""))))
    pages_with_codes = len({int(item.get("page", 0) or 0) for item in headings if item.get("page")})
    index_signals = 0

    if code_heading_count >= 12:
        index_signals += 1
    if len(sections) >= 6:
        index_signals += 1
    if len(summary_sections) >= 5:
        index_signals += 1
    if page_count and pages_with_codes / max(page_count, 1) >= 0.3:
        index_signals += 1

    return "index" if index_signals >= 2 else "general"
