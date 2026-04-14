# PDF QA Agent

本專案是一個本地 RAG（Retrieval-Augmented Generation）PDF 問答系統：

使用者上傳 PDF 後，系統會自動解析、切割、向量化，並在提問時檢索相關段落，最後交給 LLM 生成答案。

支援兩種模型：
- Gemma 4（本地，透過 Ollama）
- Gemini（雲端，透過 Google AI Studio API）
- Google ADK Agent（雲端，透過 Google Agent Development Kit + Gemini）

介面以 Gradio 建立，可在 UI 內動態切換模型。

## 系統需求
- Python 3.11+
- 建議記憶體 16GB RAM
- 已安裝 Ollama

## 前置準備
1. 安裝 Ollama
- Linux / macOS：
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
- Windows：到 https://ollama.com 下載安裝程式

2. 拉取 Gemma 4 模型
```bash
ollama pull gemma4:e4b
```
- 模型大小約 9.6GB
- 128K context window
- 16GB RAM 可流暢運行

3. 申請 Gemini API Key
- 到 Google AI Studio 申請免費 API Key

## 安裝步驟
```bash
git clone <repo>
cd pdf-qa-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# 編輯 .env 填入你的 GEMINI_API_KEY
# 如需自訂 ADK 模型，可額外設定 GOOGLE_ADK_MODEL
```

## 啟動方式
```bash
# 確認 Ollama 正在執行
ollama serve  # 如果還沒啟動的話

# 啟動應用
python app.py
# 瀏覽器開啟 http://localhost:7860
```

## Google ADK 正式用法
專案內另外提供一個符合 ADK 結構的 agent app：

- [agent.py](c:/Users/owner/Downloads/aiagent-practice-main/adk_apps/pdf_qa_agent/agent.py)
- [tools.py](c:/Users/owner/Downloads/aiagent-practice-main/adk_apps/pdf_qa_agent/tools.py)

這一組是給 `adk run` / `adk web` / `adk deploy` 使用的正式 ADK app，不是單純把 ADK 當作另一個聊天模型。

執行方式：
```bash
# 啟動 ADK Web UI
python -m google.adk.cli web adk_apps

# 互動式 CLI
python -m google.adk.cli run adk_apps/pdf_qa_agent
```

ADK agent 可用工具：
- `set_active_pdf(pdf_path)`
- `get_active_pdf()`
- `summarize_pdf(pdf_path)`
- `get_pdf_page(page_number, pdf_path=None)`
- `lookup_pdf_index(question, pdf_path=None)`
- `search_pdf(query, pdf_path=None, top_k=4)`
- `answer_pdf_question(question, pdf_path=None)`

建議的 ADK 多輪使用方式：
1. 先呼叫 `set_active_pdf("sample_natural_dialog_127.pdf")`
2. 再直接問：
   - `請整理這份文件的大綱`
   - `從這段對話判斷，談話者比較像什麼身分？`
   - `第 2 頁有什麼內容？`

ADK eval 範例資料：
- [general_pdf_eval.json](c:/Users/owner/Downloads/aiagent-practice-main/adk_apps/pdf_qa_agent/evals/general_pdf_eval.json)

目前專案裡的整合方式分成兩種：
- `Google ADK Agent (嵌入式)`：Gradio 內直接呼叫 ADK，方便快速體驗
- `adk_apps/pdf_qa_agent`：正式 ADK app 結構，適合 `run / web / eval / deploy`

## 使用說明
1. 上傳一份或多份 PDF。
2. 系統會自動進行切割與向量化，顯示「已載入 X 頁，切割為 Y 個段落」。
3. 在模型下拉選單選擇：
- Gemma 4 (本地)
- Gemini (雲端)
- Google ADK Agent
4. 在「問答模式」選擇：
- `自動`：系統自動判斷文件類型，但目前仍統一走模型推理流程
- `通用推理模式`：使用 RAG + LLM 推理，適合一般 PDF，也適用索引型 PDF
5. 若選擇 `Google ADK Agent`，系統會改用 ADK agent 與工具呼叫流程回答，優先用文件索引、頁面讀取與檢索工具完成推理。
6. 輸入問題並送出。
7. 在「最近一次檢索段落預覽」查看模型回答依據。

## ICD-11 專用驗證
如果你正在驗證 `ICD 11 Cheater VI精神、行為或神經發育障礙分類Index-20260409.pdf`，建議使用以下腳本：

```bash
# 跑完整 benchmark（Gemma + Gemini）
python benchmark_icd.py

# 跑快速回歸檢查（Gemma）
python regression_icd.py
```

`regression_icd.py` 目前會特別檢查：
- `6B85` 定義不要吃到下一個小標題
- `物質使用所致障礙有哪些？` 在模型推理下仍要抓到主要條目
- `大綱` 不要混入正文碎片
- `第 34 頁有什麼內容？` 要回完整頁面內容

### 截圖預留
- `docs/screenshots/upload-and-status.png`
- `docs/screenshots/chat-with-gemma.png`
- `docs/screenshots/chat-with-gemini.png`

## 常見問題
1. Ollama 連不上
- 請確認 `ollama serve` 有在跑

2. Gemma 4 E4B 跑太慢
- 純 CPU 推理較慢屬正常
- 可關閉其他高記憶體程式
- 若需要更高品質且有 32GB RAM，可改用 `gemma4:26b`

### 本地模型加速設定（已實作）
可在 `.env` 加入以下參數調整速度：
```env
OLLAMA_NUM_PREDICT=256
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_THREAD=7
OLLAMA_TEMPERATURE=0.2
OLLAMA_KEEP_ALIVE=30m
OLLAMA_HEALTHCHECK_TTL_SECONDS=30
ENABLE_BROAD_SEARCH=0
BROAD_SEARCH_K=8
```

說明：
- `OLLAMA_NUM_PREDICT` 越小，回覆越快（但答案可能較短）
- `OLLAMA_NUM_CTX` 降低可減少推理負擔
- `OLLAMA_NUM_THREAD` 建議設成 CPU 核心數 - 1
- `OLLAMA_KEEP_ALIVE` 可避免模型反覆卸載重載
- `ENABLE_BROAD_SEARCH=0` 可關閉額外檢索，降低每題延遲

3. Gemini API 429 錯誤
- 免費額度有 RPM 限制，等一分鐘再試

4. Google ADK Agent 無法使用
- 請確認 `.env` 內已設定 `GEMINI_API_KEY`
- `pip install -r requirements.txt` 需成功安裝 `google-adk`
- Google ADK Agent 目前使用 Gemini 作為底層模型
- 可用 `GOOGLE_ADK_MODEL` 自訂，例如 `gemini-2.5-flash`

4. 中文 PDF 亂碼
- `pdfplumber` 通常能處理
- 若仍異常，可改用 `pymupdf`

## 專案結構
```text
pdf-qa-agent/
├── app.py
├── agents/
│   ├── __init__.py
│   └── google_adk_agent.py
├── rag/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── qa_chain.py
├── models/
│   ├── __init__.py
│   ├── gemma_local.py
│   └── gemini_cloud.py
├── config.py
├── requirements.txt
├── .env.example
└── README.md
```
