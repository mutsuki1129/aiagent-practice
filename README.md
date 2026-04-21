# PDF/CSV/Web QA Agent

本專案是一個本地 RAG（Retrieval-Augmented Generation）文件問答系統。

你可以上傳 PDF / CSV，或輸入 Web URL，系統會自動：
- 解析內容
- 切段
- 建立向量索引
- 檢索證據
- 交給 LLM 生成答案

支援模型：
- Gemma 4（本地，Ollama）
- Qwen3 14B（本地，Ollama）
- Gemini（雲端）

介面使用 Gradio（`app.py`）。

---

## 目前重點能力（已實作）

- 多來源問答：PDF / CSV / Web
- 混合檢索：
  - 向量檢索（Chroma + embedding）
  - 多查詢擴展
  - RRF 融合
  - 問題貼合度重排
  - BM25 lexical 補強（可開關）
  - Cross-Encoder 重排（可開關）
- 複雜題增強：
  - 問題分解（長問題拆子問題）
  - 多步 refine 回答
  - 對話記憶壓縮注入 prompt
- 向量庫工程化：
  - collection 指紋重用（避免每次新建）
  - 自動清理舊 collection（上限可調）
  - 內建統計（cache hit / reuse / build / cleanup）
- 回歸檢查：
  - Recall@k / MRR 評估腳本
  - baseline regression 腳本
  - GitHub Actions 自動檢查 workflow

---

## 系統需求

- Python 3.11+
- 建議 16GB RAM 以上
- 已安裝 Ollama

---

## 安裝

```bash
git clone <repo>
cd aiagent-practice-main
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

> `.env` 至少要設定 `GEMINI_API_KEY`（若要用 Gemini）。

---

## Ollama 前置準備

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

若要啟用 Qwen3 14B（本地）：
```bash
ollama pull qwen3:14b
```

---

## 啟動

```bash
# 若尚未啟動 Ollama
ollama serve

# 啟動問答 UI
python app.py
# 瀏覽器開啟 http://localhost:7860
```

---

## 使用方式

1. 上傳 PDF/CSV（可多檔）或輸入 Web URL（每行一個）。
2. 按「載入檔案 / 網頁來源」。
3. 選擇模型（Gemma / Qwen / Gemini）。
4. 輸入問題送出。
5. 在「最近一次檢索段落預覽」查看證據。

定位查詢：
- PDF：`第 34 頁有什麼內容？`
- CSV：`第 10 列是什麼？` 或 `row 10`

---

## 主要設定（.env）

### 推理/檢索

```env
TOP_K=6
RETRIEVAL_PROFILE=auto
QUERY_VARIANT_COUNT=8
ENABLE_RETRIEVAL_CACHE=1
RETRIEVAL_CACHE_SIZE=128
RRF_RANK_CONSTANT=60
ENABLE_BROAD_SEARCH=0
BROAD_SEARCH_K=8
```

### 複雜問題增強

```env
COMPLEX_REASONING_PASSES=2
COMPLEX_REASONING_MIN_QUESTION_LEN=30
COMPLEX_REASONING_MIN_CONTEXT_DOCS=7
COMPLEX_RETRIEVAL_TOP_K_BOOST=2
```

### 靈活問答升級（新增）

```env
ENABLE_BM25_RETRIEVAL=1
BM25_CANDIDATE_K=24

# 預設關閉（較耗時）
ENABLE_CROSS_ENCODER_RERANK=0
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CROSS_ENCODER_TOP_K=12
```

### Ollama

```env
OLLAMA_GEMMA_MODEL=gemma4:e4b
OLLAMA_QWEN_MODEL=qwen3:14b
OLLAMA_NUM_PREDICT=512
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_THREAD=7
OLLAMA_TEMPERATURE=0.2
OLLAMA_KEEP_ALIVE=30m
```

### Chroma（collection 重用/清理）

```env
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_PREFIX=pdf_qa
CHROMA_MAX_COLLECTIONS=24
```

---

## 評估與回歸

### 1) 單次檢索評估（Recall@k / MRR）

```bash
python rag_retrieval_eval.py --cases docs/retrieval_eval_cases.example.json --k 8
```

輸出報告：
```bash
python rag_retrieval_eval.py --cases docs/retrieval_eval_cases.example.json --k 8 --report reports/rag_eval.json
```

### 2) baseline 回歸檢查

第一次建立 baseline：
```bash
python rag_retrieval_regression.py --cases docs/retrieval_eval_cases.example.json --k 8 --update-baseline
```

日常檢查：
```bash
python rag_retrieval_regression.py --cases docs/retrieval_eval_cases.example.json --k 8
```

本地一鍵入口：
```bash
python run_rag_regression.py
```

CI 等價命令：
```bash
python rag_retrieval_regression.py \
  --cases docs/retrieval_eval_cases.ci.json \
  --baseline docs/retrieval_baseline.ci.json \
  --report reports/rag_retrieval_latest_ci.json \
  --k 6 \
  --max-recall-drop 0.00 \
  --max-mrr-drop 0.00 \
  --max-error-cases 0
```

回歸腳本 exit code：
- 0：通過 / baseline 更新成功
- 2：baseline 不存在（未允許缺失）
- 3：Recall/MRR 退化超過門檻
- 4：錯誤案例數超過 `--max-error-cases`
- 5：baseline 的 k 與本次 k 不一致

---

## CI（GitHub Actions）

已提供 workflow：
- `.github/workflows/rag-retrieval-regression.yml`

觸發：
- pull_request
- push 到 main/master

內容：
- 安裝依賴
- 語法檢查
- 跑 retrieval regression（CI cases + baseline）
- 上傳報告 artifact

---

## 診斷工具

快速看檢索策略與最終 prompt：

```bash
python rag_diagnose.py --file <pdf_or_csv> --question "你的問題"
```

可搭配 `--save-prompt` 輸出 prompt 檔案。

---

## ADK Apps（獨立於 Gradio）

位置：
- `adk_apps/pdf_qa_agent/agent.py`
- `adk_apps/pdf_qa_agent/tools.py`

常用：
```bash
python -m google.adk.cli web adk_apps
python -m google.adk.cli run adk_apps/pdf_qa_agent
```

---

## 常見問題

1) Ollama 連不上
- 確認 `ollama serve` 正在執行。

2) Gemini 429
- 免費額度限制，稍後再試。

3) Web 403 / 擋頁
- 系統會自動從 `requests` fallback 到 `playwright` / `jina-reader`。
- 若仍被擋，會在來源處理狀態清楚標示。

4) 回答不夠靈活
- 先開 `ENABLE_BM25_RETRIEVAL=1`
- 再視效能開 `ENABLE_CROSS_ENCODER_RERANK=1`

---

## 專案結構（核心）

```text
.
├── app.py
├── config.py
├── requirements.txt
├── rag/
│   ├── pdf_loader.py
│   ├── csv_loader.py
│   ├── web_loader.py
│   ├── query_engine.py
│   └── vector_store.py
├── rag_diagnose.py
├── rag_retrieval_eval.py
├── rag_retrieval_regression.py
├── run_rag_regression.py
├── docs/
│   ├── retrieval_eval.md
│   ├── retrieval_eval_cases.example.json
│   ├── retrieval_eval_cases.ci.json
│   ├── retrieval_baseline.example.json
│   └── retrieval_baseline.ci.json
└── .github/workflows/
    └── rag-retrieval-regression.yml
```

---

## 操作細節附錄（保留）

以下保留較細的操作步驟，方便直接複製執行。

### A) ADK 可用工具清單（pdf_qa_agent）

- `set_active_document(file_path_or_url)`
- `get_active_document()`
- `list_available_documents(limit=20)`
- `summarize_document(file_path_or_url=None)`
- `get_document_position(position_number, file_path_or_url=None)`（PDF=頁、CSV=列、Web=段）
- `search_document(query, file_path_or_url=None, top_k=4)`
- `answer_document_question(question, file_path_or_url=None)`
- Web 便利工具：`set_active_web(url)` / `get_active_web()` / `list_active_webs(limit=20)`

### B) ADK Web UI 使用建議

1. 先送出：`set_active_web("https://example.com")`
2. 再問：`這個網頁主要在講什麼？`
3. 若剛更新 agent 程式碼，請重啟：
   - `python -m google.adk.cli web adk_apps`
4. 也支援一句話直接問：
   - `https://example.com 這個網頁主要在講什麼？`

### C) Web 抓取 403/擋頁處理細節

系統流程：
1. requests
2. 若 blocked/403 -> Playwright
3. 若仍 blocked -> jina-reader

狀態訊號：
- `web:requests`
- `web:playwright`
- `web:jina-reader`

若要啟用 Playwright：

```bash
pip install playwright
python -m playwright install chromium
```

### D) ICD-11 驗證腳本（如果你在做 ICD 驗證）

```bash
# 完整 benchmark
python benchmark_icd.py

# 快速回歸
python regression_icd.py
```

### E) 回歸檢查常用命令全集

```bash
# 1) 建 baseline（第一次）
python rag_retrieval_regression.py --cases docs/retrieval_eval_cases.example.json --k 8 --update-baseline

# 2) 本地日常回歸
python run_rag_regression.py

# 3) 嚴格模式（不允許任何 error case）
python rag_retrieval_regression.py --cases docs/retrieval_eval_cases.example.json --k 8 --max-error-cases 0

# 4) CI 等價本地跑法
python rag_retrieval_regression.py \
  --cases docs/retrieval_eval_cases.ci.json \
  --baseline docs/retrieval_baseline.ci.json \
  --report reports/rag_retrieval_latest_ci.json \
  --k 6 \
  --max-recall-drop 0.00 \
  --max-mrr-drop 0.00 \
  --max-error-cases 0
```
