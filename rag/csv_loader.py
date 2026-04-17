from __future__ import annotations

import csv
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def _clean_ocr_text(text: str) -> str:
    cleaned = (text or "").replace("\u3000", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。；：！？、）】」』])", "", cleaned)
    cleaned = re.sub(r"(?<=[（【「『])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    return cleaned


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    encodings = ("utf-8-sig", "utf-8", "cp950", "big5")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as file_obj:
                reader = csv.DictReader(file_obj)
                headers = list(reader.fieldnames or [])
                rows = [{key: _clean_ocr_text(value or "") for key, value in row.items()} for row in reader]
                return headers, rows
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise ValueError(f"CSV 讀取失敗，編碼無法辨識：{path.name}") from last_error
    raise ValueError(f"CSV 讀取失敗：{path.name}")


def load_csv_and_split(csv_path: str) -> list[Document]:
    """Load a CSV and split each row into chunked LangChain documents."""
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"找不到 CSV 檔案：{csv_path}")

    headers, rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"CSV 解析成功，但沒有可用資料列：{csv_path}")

    row_docs: list[Document] = []
    for row_index, row in enumerate(rows, start=1):
        cells: list[str] = []
        if headers:
            for header in headers:
                value = (row.get(header, "") or "").strip()
                if value:
                    cells.append(f"{header}: {value}")
        else:
            for key, value in row.items():
                key_text = (key or "").strip()
                value_text = (value or "").strip()
                if key_text or value_text:
                    cells.append(f"{key_text}: {value_text}".strip(": "))

        if not cells:
            continue

        content = "\n".join(cells)
        row_docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "page": row_index,
                    "source_type": "csv",
                    "row": row_index,
                },
            )
        )

    if not row_docs:
        raise ValueError(f"CSV 解析成功，但資料列內容皆為空白：{csv_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", ", ", " ", ""],
    )
    return splitter.split_documents(row_docs)
