from __future__ import annotations

from google.adk.agents import Agent

from config import GOOGLE_ADK_MODEL

from .tools import (
    answer_document_question,
    answer_pdf_question,
    get_active_document,
    get_active_pdf,
    get_active_web,
    get_csv_row,
    get_document_position,
    get_pdf_page,
    list_available_csvs,
    list_available_documents,
    list_available_pdfs,
    list_active_webs,
    lookup_document_index,
    lookup_pdf_index,
    search_document,
    search_pdf,
    set_active_csv,
    set_active_document,
    set_active_pdf,
    set_active_web,
    summarize_document,
    summarize_pdf,
)


root_agent = Agent(
    name="pdf_qa_agent",
    model=GOOGLE_ADK_MODEL,
    description="A tool-using agent for summarizing and querying PDF/CSV files and web URLs.",
    instruction=(
        "你是文件問答 ADK Agent（支援 PDF/CSV/Web URL），請使用繁體中文回答。\n"
        "你的任務是協助使用者理解本機文件與網頁內容，並善用工具完成推理。\n"
        "你不可以直接宣稱『只能總結 PDF/CSV』；遇到網址一定要先呼叫 Web 工具。\n"
        "工具使用規則：\n"
        "0. 如果本輪訊息包含檔名、附件或網址，第一步先用 set_active_document(該來源)。\n"
        "0-1. 若文字含有 http:// 或 https://，優先呼叫 set_active_web(url)，再呼叫 answer_document_question。\n"
        "0-2. 若問題中同時包含 URL 與提問（例如：`https://example.com 主要在講什麼？`），先抽出 URL 設為 active，再回答問題。\n"
        "1. 當使用者問文件大意、摘要、大綱、主要在講什麼，優先使用 summarize_document。\n"
        "2. 當使用者問指定位置內容，使用 get_document_position（PDF=頁；CSV=列；Web=段）。\n"
        "3. 當使用者問代碼、分類、頁碼對照且檔案為索引型 PDF，可使用 lookup_document_index。\n"
        "4. 其他一般問題，先用 answer_document_question；必要時補充 search_document。\n"
        "5. 若使用者想連續追問同一份文件，先用 set_active_document 設定 active 文件。\n"
        "6. 若是網址優先用 set_active_web；若不確定目前來源，先用 list_available_documents / list_active_webs 與 get_active_document。\n"
        "7. 若遇到「目前有多份文件，尚未設定 active 文件」錯誤，必須先 set_active_document 再回答。\n"
        "8. 若工具已回傳足夠證據，請直接整理成完整答案，不要只回原始片段。\n"
        "9. 不要預設所有文件都是 ICD；必須依工具結果判斷。"
    ),
    tools=[
        set_active_document,
        get_active_document,
        list_available_documents,
        set_active_web,
        get_active_web,
        list_active_webs,
        summarize_document,
        get_document_position,
        lookup_document_index,
        search_document,
        answer_document_question,
        set_active_csv,
        get_csv_row,
        list_available_csvs,
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
