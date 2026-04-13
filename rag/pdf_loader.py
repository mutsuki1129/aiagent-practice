from __future__ import annotations

from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def load_and_split(pdf_path: str) -> list[Document]:
    """Load a PDF with pdfplumber and split into chunked LangChain documents."""
    path = Path(pdf_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"找不到 PDF 檔案：{pdf_path}")

    docs: list[Document] = []
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_index},
                )
            )

    if not docs:
        raise ValueError(f"PDF 解析成功，但未擷取到可用文字：{pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    return splitter.split_documents(docs)
