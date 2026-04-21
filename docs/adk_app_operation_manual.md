# ADK 與 app.py 操作手冊

本手冊依據 `README.md` 與目前程式實作整理，聚焦兩個入口：
1) Gradio 介面（`app.py`）
2) Google ADK App（`adk_apps/pdf_qa_agent`）

---

## 1. 適用範圍

- 本地 RAG 文件問答（PDF / CSV / Web URL）
- 模型來源：
  - 本地 Ollama：Gemma 4、Qwen3 14B
  - 雲端 Gemini（需 API Key）

---

## 2. 環境準備

### 2.1 安裝

```bash
git clone <repo>
cd aiagent-practice-main
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2.2 必要設定

`.env` 建議至少確認：

```env
GEMINI_API_KEY=your_key_here            # 若要使用 Gemini
GOOGLE_ADK_MODEL=gemini-2.5-flash       # ADK 模型
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_GEMMA_MODEL=gemma4:e4b
OLLAMA_QWEN_MODEL=qwen3:14b
```

### 2.3 Ollama 準備

```bash
ollama pull gemma4:e4b
ollama pull qwen3:14b
ollama serve
```

---

## 3. app.py（Gradio UI）操作

### 3.1 啟動

```bash
python app.py
```

- 預設服務：`http://localhost:7860`
- 程式綁定：`0.0.0.0:7860`

### 3.2 UI 元件與用途

左側：
- 上傳檔案（PDF / CSV，可多檔）
- 網頁 URL（可多行）
- `載入檔案 / 網頁來源`
- 模型選擇：Gemma / Qwen / Gemini
- 問答模式：`自動` 或 `通用推理模式`

右側：
- 對話區
- 問題輸入框與送出
- 清除對話
- 最近一次檢索段落預覽

### 3.3 標準操作流程

1. 上傳 PDF/CSV 或貼上 URL（可多行）。
2. 點 `載入檔案 / 網頁來源`。
3. 觀察狀態：
   - 模型狀態（Ollama/Gemini 可用性）
   - 文件模式狀態
4. 選擇模型（本地或雲端）。
5. 提問並查看：
   - 回答內容
   - `最近一次檢索段落預覽`（證據）

### 3.4 建議提問範例

- PDF：`第 34 頁有什麼內容？`
- CSV：`第 10 列是什麼？` / `row 10`
- Web：`這個網頁主要在講什麼？`
- 索引型文件：`某代碼屬於哪一類？`

### 3.5 常用檢索/推理參數（.env）

```env
TOP_K=6
RETRIEVAL_PROFILE=auto
QUERY_VARIANT_COUNT=8
ENABLE_RETRIEVAL_CACHE=1
RETRIEVAL_CACHE_SIZE=128
RRF_RANK_CONSTANT=60
ENABLE_BM25_RETRIEVAL=1
BM25_CANDIDATE_K=24
ENABLE_CROSS_ENCODER_RERANK=0
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CROSS_ENCODER_TOP_K=12
```

調校建議：
- 先開 `ENABLE_BM25_RETRIEVAL=1`（靈活度提升通常明顯）。
- 再視效能開 `ENABLE_CROSS_ENCODER_RERANK=1`（品質更好但較慢）。

### 3.6 app.py 常見問題

1) Ollama 無法連線
- 確認 `ollama serve` 是否啟動。
- 確認 `.env` 的 `OLLAMA_BASE_URL`。

2) Gemini 429 / 配額不足
- 稍後重試或改用本地模型。

3) Web 403 / 擋頁
- 系統會嘗試 `requests -> playwright -> jina-reader`。
- 若要啟用 Playwright：
  ```bash
  pip install playwright
  python -m playwright install chromium
  ```

---

## 4. ADK（adk_apps/pdf_qa_agent）操作

### 4.1 位置

- `adk_apps/pdf_qa_agent/agent.py`
- `adk_apps/pdf_qa_agent/tools.py`

### 4.2 啟動方式

1) 啟動 ADK Web UI
```bash
python -m google.adk.cli web adk_apps
```

2) 直接 run agent
```bash
python -m google.adk.cli run adk_apps/pdf_qa_agent
```

### 4.3 ADK 工具（README 對齊）

- `set_active_document(file_path_or_url)`
- `get_active_document()`
- `list_available_documents(limit=20)`
- `summarize_document(file_path_or_url=None)`
- `get_document_position(position_number, file_path_or_url=None)`
  - PDF = 頁、CSV = 列、Web = 段
- `search_document(query, file_path_or_url=None, top_k=4)`
- `answer_document_question(question, file_path_or_url=None)`
- Web 便利工具：
  - `set_active_web(url)`
  - `get_active_web()`
  - `list_active_webs(limit=20)`

### 4.4 ADK 推薦操作流程

情境 A：先指定來源再問
1. `set_active_document("<你的檔案路徑或URL>")`
2. `summarize_document()` 或 `answer_document_question("你的問題")`

情境 B：Web 問答
1. `set_active_web("https://example.com")`
2. `answer_document_question("這個網頁主要在講什麼？")`

情境 C：定位查詢
- `get_document_position(34)`（依來源自動解讀為頁/列/段）

### 4.5 ADK 注意事項

- 若出現「多份文件但未設定 active 文件」，先執行 `set_active_document(...)`。
- URL 問題建議先 `set_active_web(url)`，再問問題。
- 工具已回傳證據時，應整理成答案，不只貼原文片段。

---

## 5. 快速驗證清單

### app.py
- [ ] `python app.py` 可啟動
- [ ] 可成功載入至少一份 PDF/CSV 或一個 URL
- [ ] 可完成一輪問答且有檢索預覽

### ADK
- [ ] `python -m google.adk.cli web adk_apps` 可啟動
- [ ] `set_active_document(...)` 可成功
- [ ] `answer_document_question(...)` 可產生回答

---
