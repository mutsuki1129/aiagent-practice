from __future__ import annotations

from google.adk.agents import Agent

from config import GOOGLE_ADK_MODEL

from .tools import (
    answer_pdf_question,
    get_active_pdf,
    get_pdf_page,
    lookup_pdf_index,
    search_pdf,
    set_active_pdf,
    summarize_pdf,
)


root_agent = Agent(
    name="pdf_qa_agent",
    model=GOOGLE_ADK_MODEL,
    description="A tool-using agent for summarizing and querying local PDF files.",
    instruction=(
        "你是 PDF 問答 ADK Agent，請使用繁體中文回答。\n"
        "你的任務是協助使用者理解本機 PDF 文件內容，並善用工具完成推理。\n"
        "工作原則如下：\n"
        "1. 當使用者問文件大意、摘要、大綱、主要在講什麼，優先使用 summarize_pdf。\n"
        "2. 當使用者問第幾頁內容，優先使用 get_pdf_page。\n"
        "3. 當使用者問代碼、分類、條目、頁碼等結構化問題，先嘗試 lookup_pdf_index。\n"
        "4. 若需要更多上下文或一般語意搜尋，使用 search_pdf 或 answer_pdf_question。\n"
        "5. 若使用者想連續追問同一份 PDF，可以先用 set_active_pdf 設定 active PDF，後續工具就能省略 pdf_path。\n"
        "6. 回答時要明確說明依據，盡量附上頁碼；若工具已提供完整清單，不要漏掉項目。\n"
        "7. 不要假設所有 PDF 都是 ICD 文件；必須根據工具回傳內容判斷。"
    ),
    tools=[
        set_active_pdf,
        get_active_pdf,
        summarize_pdf,
        get_pdf_page,
        lookup_pdf_index,
        search_pdf,
        answer_pdf_question,
    ],
)
