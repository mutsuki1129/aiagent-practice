from __future__ import annotations

from html import unescape
import re
from typing import Any

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

_BLOCKED_MARKERS = (
    "access denied",
    "you don't have permission to access",
    "forbidden",
    "403 forbidden",
    "request blocked",
    "service unavailable",
    "attention required",
    "cloudflare",
    "verify you are human",
    "captcha",
    "robot check",
    "bot detection",
    "incapsula",
    "akamai",
    "please enable javascript",
    "enable cookies",
)


def _extract_readable_text(html: str) -> str:
    text = html or ""
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_structured_hints(html: str) -> str:
    source = html or ""
    if not source:
        return ""

    hints: list[str] = []

    title_match = re.search(r'(?is)<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']', source)
    if not title_match:
        title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", source)
    if title_match:
        title = re.sub(r"\s+", " ", unescape(title_match.group(1))).strip()
        if title:
            hints.append(f"title: {title}")

    desc_match = re.search(r'(?is)<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', source)
    if not desc_match:
        desc_match = re.search(r'(?is)<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']', source)
    if desc_match:
        desc = re.sub(r"\s+", " ", unescape(desc_match.group(1))).strip()
        if desc:
            hints.append(f"description: {desc}")

    article_match = re.search(r"(?is)<article[^>]*>(.*?)</article>", source)
    if article_match:
        article_text = _extract_readable_text(article_match.group(1))
        if article_text:
            hints.append(article_text[:4000])

    return "\n".join(hints).strip()


def _looks_blocked_text(text: str) -> tuple[bool, str]:
    lowered = (text or "").strip().lower()
    if not lowered:
        return True, "empty"
    sample = lowered[:1800]
    for marker in _BLOCKED_MARKERS:
        if marker in sample:
            return True, marker
    return False, ""


def _fetch_with_requests(target: str, timeout: int) -> tuple[str, dict[str, Any], int]:
    with requests.Session() as session:
        response = session.get(
            target,
            timeout=timeout,
            headers=_BROWSER_HEADERS,
            allow_redirects=True,
        )

    content_type = (response.headers.get("content-type", "") or "").lower()
    if "text/html" in content_type or "application/xhtml+xml" in content_type:
        readable = _extract_readable_text(response.text)
        structured = _extract_structured_hints(response.text)
        text = "\n".join([part for part in (structured, readable) if part]).strip()
    else:
        text = re.sub(r"\s+", " ", response.text or "").strip()

    return text, dict(response.headers), int(response.status_code)


def _fetch_with_playwright(target: str, timeout: int) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "requests 取得網頁受限，且未安裝 Playwright。"
            "請安裝：pip install playwright，並執行：python -m playwright install chromium"
        ) from exc

    timeout_ms = max(timeout, 10) * 1000
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        context = browser.new_context(
            user_agent=_BROWSER_HEADERS["User-Agent"],
            locale="zh-TW",
            extra_http_headers={
                "Accept": _BROWSER_HEADERS["Accept"],
                "Accept-Language": _BROWSER_HEADERS["Accept-Language"],
            },
        )
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        page = context.new_page()
        page.goto(target, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 15000))
        except Exception:
            pass

        body_text = page.inner_text("body")
        html = page.content()
        structured = _extract_structured_hints(html)
        if body_text:
            body_clean = re.sub(r"\s+", " ", body_text).strip()
            text = "\n".join([part for part in (structured, body_clean) if part]).strip()
        else:
            text = "\n".join([part for part in (structured, _extract_readable_text(html)) if part]).strip()

        context.close()
        browser.close()
        return text


def _fetch_with_jina_reader(target: str, timeout: int) -> str:
    reader_url = f"https://r.jina.ai/http://{target.replace('https://', '').replace('http://', '')}"
    response = requests.get(reader_url, headers=_BROWSER_HEADERS, timeout=max(timeout, 20))
    response.raise_for_status()
    return re.sub(r"\s+", " ", response.text or "").strip()


def load_web_and_split(url: str, timeout: int = 15) -> list[Document]:
    """Load a web page and split into chunked LangChain documents.

    Strategy:
    1) requests with browser-like headers
    2) fallback to Playwright when blocked (403/anti-bot) or content quality is too low
    3) fallback to jina-ai reader mirror when still blocked
    """
    target = (url or "").strip()
    if not target:
        raise ValueError("網址不可為空。")
    if not target.lower().startswith(("http://", "https://")):
        raise ValueError(f"不支援的網址格式：{target}")

    page_text = ""
    status_code = 0
    fetch_method = "requests"
    blocked_marker = ""

    try:
        page_text, _, status_code = _fetch_with_requests(target, timeout=timeout)
    except requests.RequestException:
        status_code = 0

    blocked_detected, blocked_marker = _looks_blocked_text(page_text)
    need_fallback = status_code in {0, 401, 403, 406, 429, 503} or len(page_text) < 120 or blocked_detected
    if need_fallback:
        try:
            page_text = _fetch_with_playwright(target, timeout=timeout)
            fetch_method = "playwright"
            blocked_detected, blocked_marker = _looks_blocked_text(page_text)
        except Exception as exc:
            if len(page_text) < 30:
                raise RuntimeError(f"網頁載入失敗（{target}）：{exc}") from exc

    if blocked_detected:
        try:
            page_text = _fetch_with_jina_reader(target, timeout=timeout)
            fetch_method = "jina-reader"
            blocked_detected, blocked_marker = _looks_blocked_text(page_text)
        except Exception:
            pass

    if blocked_detected:
        raise RuntimeError(
            f"網站拒絕存取或返回擋頁（偵測字詞：{blocked_marker or 'unknown'}）。"
            "可改用可公開讀取的文章網址、登入後可讀版本，或稍後重試。"
        )

    if len(page_text) < 30:
        raise ValueError(f"網頁內容過短或無法擷取文字：{target}")

    base_doc = Document(
        page_content=page_text,
        metadata={
            "source": target,
            "page": 1,
            "source_type": "web",
            "fetch_method": fetch_method,
            "status_code": status_code,
            "blocked_marker": blocked_marker,
        },
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", ". ", ", ", " ", ""],
    )
    chunks = splitter.split_documents([base_doc])
    for idx, doc in enumerate(chunks, start=1):
        doc.metadata["page"] = idx
        doc.metadata["source_type"] = "web"
        doc.metadata["source"] = target
        doc.metadata["fetch_method"] = fetch_method
        doc.metadata["status_code"] = status_code
        doc.metadata["blocked_marker"] = blocked_marker
    return chunks
