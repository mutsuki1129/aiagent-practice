from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import requests

from config import GEMINI_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_K
from models.gemma_local import get_gemma_llm
from models.gemini_cloud import get_gemini_llm
from rag.pdf_loader import load_and_split
from rag.qa_chain import build_qa_chain
from rag.vector_store import create_vector_store, get_retriever


def check_model_status() -> str:
    """Return markdown status for Ollama and Gemini availability."""
    ollama_status = "不可用"
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.ok:
            models = {m.get("name", "") for m in response.json().get("models", []) if isinstance(m, dict)}
            if OLLAMA_MODEL in models:
                ollama_status = f"已連線（{OLLAMA_MODEL} 可用）"
            else:
                ollama_status = f"已連線（但缺少模型 {OLLAMA_MODEL}）"
        else:
            ollama_status = "服務異常"
    except requests.RequestException:
        ollama_status = "未連線（請確認 ollama serve）"

    gemini_status = "API Key 已設定" if GEMINI_API_KEY.strip() else "未設定 GEMINI_API_KEY"

    return (
        "### 模型狀態\n"
        f"- Ollama / Gemma 4：{ollama_status}\n"
        f"- Gemini：{gemini_status}"
    )


def _format_retrieved_preview(source_docs: list[Any]) -> str:
    if not source_docs:
        return "尚無檢索結果。"

    lines: list[str] = []
    for idx, doc in enumerate(source_docs[:4], start=1):
        source = doc.metadata.get("source", "未知檔案")
        page = doc.metadata.get("page", "?")
        content = (doc.page_content or "").strip().replace("\n", " ")
        if len(content) > 220:
            content = content[:220] + "..."
        lines.append(f"{idx}. [{source} - 第 {page} 頁] {content}")

    return "\n\n".join(lines)


def process_uploaded_pdfs(files: list[str] | None):
    if not files:
        return (
            None,
            "請先上傳至少一份 PDF。",
            check_model_status(),
            "尚無檢索結果。",
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
        )

    all_docs = []
    unique_pages: set[tuple[str, int]] = set()
    used_files: list[str] = []

    try:
        for file_path in files:
            docs = load_and_split(file_path)
            all_docs.extend(docs)
            used_files.append(Path(file_path).name)
            for doc in docs:
                source = str(doc.metadata.get("source", ""))
                page = int(doc.metadata.get("page", 0))
                if source and page > 0:
                    unique_pages.add((source, page))

        vector_store = create_vector_store(all_docs)
        retriever = get_retriever(vector_store, top_k=TOP_K)

        status = (
            f"已載入 {len(unique_pages)} 頁，切割為 {len(all_docs)} 個段落。"
            f"\n檔案：{', '.join(used_files)}"
        )
        return (
            retriever,
            status,
            check_model_status(),
            "尚無檢索結果。",
            gr.update(interactive=True),
            gr.update(interactive=True),
            [],
        )
    except Exception as exc:
        return (
            None,
            f"PDF 處理失敗：{exc}",
            check_model_status(),
            "尚無檢索結果。",
            gr.update(interactive=False),
            gr.update(interactive=False),
            [],
        )


def ask_question(
    user_question: str,
    chat_history: list[dict[str, str]] | None,
    retriever,
    model_choice: str,
    conversation_state: list[tuple[str, str]] | None,
):
    history = chat_history or []
    conversations = conversation_state or []

    if not user_question or not user_question.strip():
        return history, "", check_model_status(), "請輸入問題。", conversations

    if retriever is None:
        history.append({"role": "assistant", "content": "請先上傳並處理 PDF，完成後再開始提問。"})
        return history, "", check_model_status(), "尚無檢索結果。", conversations

    try:
        llm = get_gemma_llm() if model_choice.startswith("Gemma") else get_gemini_llm()
        qa_chain = build_qa_chain(retriever, llm)
        result = qa_chain.invoke({"query": user_question.strip()})

        answer = str(result.get("result", "")).strip() or "根據文件內容，我找不到相關資訊"
        source_docs = result.get("source_documents", [])
        preview = _format_retrieved_preview(source_docs)

        pages = sorted(
            {
                f"第 {doc.metadata.get('page', '?')} 頁"
                for doc in source_docs
                if doc.metadata.get("page") is not None
            }
        )
        if pages:
            answer = f"{answer}\n\n參考來源：{'、'.join(pages)}"

        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": answer})
        conversations.append((user_question, answer))
        return history, "", check_model_status(), preview, conversations
    except Exception as exc:
        history.append({"role": "assistant", "content": f"處理問題時發生錯誤：{exc}"})
        return history, "", check_model_status(), "檢索失敗，請確認模型或文件狀態。", conversations


def clear_chat():
    return [], "", "尚無檢索結果。", []


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PDF 智能問答系統") as demo:
        retriever_state = gr.State(value=None)
        conversation_state = gr.State(value=[])

        gr.Markdown("# 📄 PDF 智能問答系統")
        gr.Markdown("支援 Gemma 4 本地模型（Ollama）與 Gemini 雲端模型，採用 RAG 檢索增強生成。")

        with gr.Row():
            with gr.Column(scale=1):
                uploader = gr.File(
                    label="上傳 PDF（可多檔）",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )
                model_choice = gr.Dropdown(
                    label="模型選擇",
                    choices=["Gemma 4 (本地)", "Gemini (雲端)"],
                    value="Gemma 4 (本地)",
                )
                process_status = gr.Markdown("尚未載入 PDF。")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="對話區", height=480)
                with gr.Row():
                    user_input = gr.Textbox(
                        label="輸入問題",
                        placeholder="請輸入你想問 PDF 的問題...",
                        interactive=False,
                    )
                    send_btn = gr.Button("送出", variant="primary", interactive=False)
                clear_btn = gr.Button("清除對話")

        model_status = gr.Markdown(check_model_status())

        with gr.Accordion("最近一次檢索段落預覽", open=False):
            preview = gr.Markdown("尚無檢索結果。")

        uploader.change(
            fn=process_uploaded_pdfs,
            inputs=[uploader],
            outputs=[
                retriever_state,
                process_status,
                model_status,
                preview,
                user_input,
                send_btn,
                conversation_state,
            ],
        )

        send_btn.click(
            fn=ask_question,
            inputs=[user_input, chatbot, retriever_state, model_choice, conversation_state],
            outputs=[chatbot, user_input, model_status, preview, conversation_state],
        )
        user_input.submit(
            fn=ask_question,
            inputs=[user_input, chatbot, retriever_state, model_choice, conversation_state],
            outputs=[chatbot, user_input, model_status, preview, conversation_state],
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, user_input, preview, conversation_state],
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
