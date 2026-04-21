from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG diagnostic runner (no UI)")
    parser.add_argument("--file", action="append", dest="files", default=[], help="PDF/CSV file path (repeatable)")
    parser.add_argument("--url", action="append", dest="urls", default=[], help="Web URL (repeatable)")
    parser.add_argument("--question", required=True, help="Question to diagnose")
    parser.add_argument("--model", default="Gemma 4 (本地)", help="Model choice label")
    parser.add_argument("--mode", default="通用推理模式", help="Mode choice label")
    parser.add_argument("--top", type=int, default=5, help="How many retrieved chunks to print")
    parser.add_argument("--save-prompt", default="", help="Optional path to save final prompt")
    return parser.parse_args()


def _safe_text(text: str, limit: int = 700) -> str:
    value = (text or "").strip().replace("\r\n", "\n")
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _resolve_strategy(question: str, mode_choice: str, detected_mode: str) -> dict[str, Any]:
    resolved_query_mode = app._resolve_query_mode(mode_choice, detected_mode, question)
    row_lookup = app._is_row_query(question)
    page_lookup = resolved_query_mode == "page_lookup" or row_lookup
    list_query = resolved_query_mode == "list"
    lookup_query = resolved_query_mode == "lookup"
    inference_query = resolved_query_mode == "inference"
    summary_query = resolved_query_mode == "summary"

    retrieval_top_k = 5
    if resolved_query_mode == "list":
        retrieval_top_k = 9
    elif resolved_query_mode == "inference":
        retrieval_top_k = 7
    elif resolved_query_mode == "summary":
        retrieval_top_k = 8

    return {
        "resolved_query_mode": resolved_query_mode,
        "row_lookup": row_lookup,
        "page_lookup": page_lookup,
        "list_query": list_query,
        "lookup_query": lookup_query,
        "inference_query": inference_query,
        "summary_query": summary_query,
        "retrieval_top_k": retrieval_top_k,
    }


def main() -> int:
    args = _parse_args()
    files = [str(Path(p)) for p in args.files]
    urls_text = "\n".join(args.urls)

    (
        retriever,
        document_index,
        process_status,
        _model_status,
        _doc_mode_status,
        _preview,
        _user_input,
        _send_btn,
        all_docs,
        _conversation,
        detected_mode,
    ) = app.process_uploaded_sources(files=files, web_urls_text=urls_text)

    if retriever is None:
        print("[ERROR] source processing failed")
        print(process_status)
        return 1

    question = args.question.strip()
    strategy = _resolve_strategy(question, args.mode, detected_mode)

    resolved_profile = app._resolve_retrieval_profile(
        question,
        all_docs=all_docs,
        is_summary=strategy["summary_query"],
        is_list_query=strategy["list_query"],
        is_lookup_query=strategy["lookup_query"],
        is_page_lookup=strategy["page_lookup"],
        is_inference_query=strategy["inference_query"],
    )

    context_docs: list[Any] = []
    candidate_hint = ""

    if strategy["lookup_query"]:
        context_docs = app._select_reasoning_docs(
            question,
            all_docs,
            retriever,
            document_index,
            top_k=max(5, strategy["retrieval_top_k"]),
            retrieval_profile=resolved_profile,
        )
        if app._is_index_document_mode(detected_mode):
            candidate_hint = app.answer_with_document_index(question, document_index)

    if not context_docs:
        context_docs = app._select_reasoning_docs(
            question,
            all_docs,
            retriever,
            document_index,
            top_k=strategy["retrieval_top_k"],
            retrieval_profile=resolved_profile,
        )

    prompt_docs = app._prepare_reasoning_context(
        question,
        context_docs,
        is_list_query=strategy["list_query"],
        is_lookup_query=strategy["lookup_query"],
        is_page_lookup=strategy["page_lookup"],
    )

    final_prompt = app._build_reasoning_prompt(
        question,
        prompt_docs,
        document_index,
        detected_mode,
        candidate_hint,
        resolved_query_mode=strategy["resolved_query_mode"],
        is_local_model=app._is_local_ollama_choice(args.model),
    )

    print("=== RAG DIAGNOSIS ===")
    print(f"question: {question}")
    print(f"detected_mode: {detected_mode}")
    print(f"resolved_query_mode: {strategy['resolved_query_mode']}")
    print(f"resolved_profile: {resolved_profile}")
    print(f"retrieval_top_k: {strategy['retrieval_top_k']}")
    print(f"retrieved_docs: {len(context_docs)}")

    print("\n=== TOP RETRIEVED CHUNKS ===")
    for idx, doc in enumerate(context_docs[: max(1, args.top)], start=1):
        source = str(doc.metadata.get("source", "未知來源") or "未知來源")
        position = app._format_position_label(doc)
        snippet = _safe_text(str(getattr(doc, "page_content", "") or ""), limit=800)
        print(f"[{idx}] {source} | {position}")
        print(snippet)
        print("-" * 80)

    print("\n=== FINAL PROMPT TO LLM ===")
    print(final_prompt)

    if args.save_prompt.strip():
        output_path = Path(args.save_prompt).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_prompt, encoding="utf-8")
        print(f"\n[prompt saved] {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
