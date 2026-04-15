from __future__ import annotations

from google.adk.agents import Agent

from config import GOOGLE_ADK_MODEL

from .tools import (
    answer_pdf_question,
    get_active_pdf,
    get_pdf_page,
    list_available_pdfs,
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
        "工具使用規則：\n"
        "0. 如果本輪訊息包含檔名或附件，第一步先用 set_active_pdf(該檔名)；若未指定也要嘗試由附件自動判斷。\n"
        "1. 當使用者問文件大意、摘要、大綱、主要在講什麼，優先使用 summarize_pdf。\n"
        "2. 當使用者問指定頁面內容，使用 get_pdf_page。\n"
        "3. 當使用者問代碼、分類、頁碼對照，使用 lookup_pdf_index。\n"
        "4. 其他一般問題，先用 answer_pdf_question；必要時可補充 search_pdf。\n"
        "5. 若使用者想連續追問同一份 PDF，先用 set_active_pdf 設定 active PDF，後續可省略 pdf_path。\n"
        "6. 若不確定目前可用檔案，先用 list_available_pdfs 與 get_active_pdf。\n"
        "7. 若遇到『目前有多份 PDF，尚未設定 active PDF』錯誤，必須先 set_active_pdf 再繼續回答。\n"
        "8. 若工具已回傳足夠證據，請直接整理成完整答案，不要只回原始片段。\n"
        "9. 不要預設所有 PDF 都是 ICD；必須依工具結果判斷。"
    ),
    tools=[
        set_active_pdf,
        get_active_pdf,
        list_available_pdfs,
        summarize_pdf,
        get_pdf_page,
        lookup_pdf_index,
        search_pdf,
        answer_pdf_question,
    ],
)
