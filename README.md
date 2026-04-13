# PDF QA Agent

本專案是一個本地 RAG（Retrieval-Augmented Generation）PDF 問答系統：

使用者上傳 PDF 後，系統會自動解析、切割、向量化，並在提問時檢索相關段落，最後交給 LLM 生成答案。

支援兩種模型：
- Gemma 4（本地，透過 Ollama）
- Gemini（雲端，透過 Google AI Studio API）

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
```

## 啟動方式
```bash
# 確認 Ollama 正在執行
ollama serve  # 如果還沒啟動的話

# 啟動應用
python app.py
# 瀏覽器開啟 http://localhost:7860
```

## 使用說明
1. 上傳一份或多份 PDF。
2. 系統會自動進行切割與向量化，顯示「已載入 X 頁，切割為 Y 個段落」。
3. 在模型下拉選單選擇：
- Gemma 4 (本地)
- Gemini (雲端)
4. 輸入問題並送出。
5. 在「最近一次檢索段落預覽」查看模型回答依據。

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

3. Gemini API 429 錯誤
- 免費額度有 RPM 限制，等一分鐘再試

4. 中文 PDF 亂碼
- `pdfplumber` 通常能處理
- 若仍異常，可改用 `pymupdf`

## 專案結構
```text
pdf-qa-agent/
├── app.py
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
