from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality with Recall@k and MRR.")
    parser.add_argument(
        "--cases",
        required=True,
        help="Path to eval cases JSON. See docs/retrieval_eval_cases.example.json",
    )
    parser.add_argument("--k", type=int, default=8, help="Top-k for Recall@k / MRR")
    parser.add_argument("--mode", default="通用推理模式", help="Mode choice label")
    parser.add_argument("--report", default="", help="Optional output report JSON path")
    return parser.parse_args()


def load_cases(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("cases JSON must be a list")
    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        case = dict(item)
        case.setdefault("name", f"case_{idx}")
        case.setdefault("question", "")
        case.setdefault("expected_pages", [])
        case.setdefault("expected_source_contains", [])
        case.setdefault("expected_terms", [])
        cases.append(case)
    return cases


def _load_cases(path: str) -> list[dict[str, Any]]:
    # Backward-compatible alias.
    return load_cases(path)


def _resolve_sources(case: dict[str, Any]) -> tuple[list[str], str]:
    files: list[str] = []
    for key in ("file_path", "pdf_path", "csv_path"):
        value = str(case.get(key, "") or "").strip()
        if value:
            files.append(value)
    urls = case.get("urls") or []
    if isinstance(urls, str):
        urls = [urls]
    url_lines = [str(url).strip() for url in urls if str(url).strip()]
    return files, "\n".join(url_lines)


def _is_relevant(doc: Any, case: dict[str, Any]) -> bool:
    expected_pages = [int(x) for x in case.get("expected_pages", []) if str(x).strip().isdigit()]
    expected_sources = [str(x).strip().lower() for x in case.get("expected_source_contains", []) if str(x).strip()]
    expected_terms = [str(x).strip().lower() for x in case.get("expected_terms", []) if str(x).strip()]

    page = int(doc.metadata.get("page", 0) or 0)
    source = str(doc.metadata.get("source", "") or "").lower()
    content = str(getattr(doc, "page_content", "") or "").lower()

    if expected_pages and page not in expected_pages:
        return False
    if expected_sources and not any(term in source for term in expected_sources):
        return False
    if expected_terms and not any(term in content for term in expected_terms):
        return False
    return bool(expected_pages or expected_sources or expected_terms)


def _rank_first_relevant(docs: list[Any], case: dict[str, Any], k: int) -> int:
    for rank, doc in enumerate(docs[: max(1, k)], start=1):
        if _is_relevant(doc, case):
            return rank
    return 0


def _resolve_query_mode(question: str, mode_choice: str, detected_mode: str) -> dict[str, Any]:
    resolved_query_mode = app._resolve_query_mode(mode_choice, detected_mode, question)
    row_lookup = app._is_row_query(question)
    page_lookup = resolved_query_mode == "page_lookup" or row_lookup
    list_query = resolved_query_mode == "list"
    lookup_query = resolved_query_mode == "lookup"
    inference_query = resolved_query_mode == "inference"
    summary_query = resolved_query_mode == "summary"

    retrieval_top_k = 7
    if resolved_query_mode == "list":
        retrieval_top_k = 12
    elif resolved_query_mode == "inference":
        retrieval_top_k = 10
    elif resolved_query_mode == "summary":
        retrieval_top_k = 10
    elif resolved_query_mode == "lookup":
        retrieval_top_k = 8

    if app._is_complex_reasoning_query(
        question,
        resolved_query_mode=resolved_query_mode,
        context_doc_count=0,
    ):
        retrieval_top_k += max(0, app.COMPLEX_RETRIEVAL_TOP_K_BOOST)

    return {
        "resolved_query_mode": resolved_query_mode,
        "page_lookup": page_lookup,
        "list_query": list_query,
        "lookup_query": lookup_query,
        "inference_query": inference_query,
        "summary_query": summary_query,
        "retrieval_top_k": retrieval_top_k,
    }


def evaluate_case(case: dict[str, Any], *, k: int, mode_choice: str) -> dict[str, Any]:
    question = str(case.get("question", "") or "").strip()
    if not question:
        raise ValueError(f"case {case.get('name')} missing question")

    files, urls_text = _resolve_sources(case)

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
        return {
            "name": case.get("name", "unknown"),
            "question": question,
            "error": process_status,
            "rank": 0,
            "hit": False,
            "rr": 0.0,
        }

    mode_info = _resolve_query_mode(question, mode_choice, detected_mode)
    resolved_profile = app._resolve_retrieval_profile(
        question,
        all_docs=all_docs,
        is_summary=mode_info["summary_query"],
        is_list_query=mode_info["list_query"],
        is_lookup_query=mode_info["lookup_query"],
        is_page_lookup=mode_info["page_lookup"],
        is_inference_query=mode_info["inference_query"],
    )

    docs = app._select_reasoning_docs(
        question,
        all_docs,
        retriever,
        document_index,
        top_k=max(k, mode_info["retrieval_top_k"]),
        retrieval_profile=resolved_profile,
    )

    rank = _rank_first_relevant(docs, case, k)
    return {
        "name": case.get("name", "unknown"),
        "question": question,
        "rank": rank,
        "hit": rank > 0,
        "rr": (1.0 / rank) if rank > 0 else 0.0,
        "retrieved": len(docs),
        "resolved_profile": resolved_profile,
        "resolved_query_mode": mode_info["resolved_query_mode"],
    }


def run_retrieval_eval(cases: list[dict[str, Any]], *, k: int, mode_choice: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            result = evaluate_case(case, k=k, mode_choice=mode_choice)
        except Exception as exc:
            result = {
                "name": case.get("name", "unknown"),
                "question": case.get("question", ""),
                "error": str(exc),
                "rank": 0,
                "hit": False,
                "rr": 0.0,
            }
        results.append(result)

    valid = [item for item in results if not item.get("error")]
    hit_count = sum(1 for item in valid if item.get("hit"))
    recall_at_k = (hit_count / len(valid)) if valid else 0.0
    mrr = (sum(float(item.get("rr", 0.0)) for item in valid) / len(valid)) if valid else 0.0

    return {
        "k": k,
        "case_count": len(results),
        "valid_count": len(valid),
        "recall_at_k": recall_at_k,
        "mrr_at_k": mrr,
        "results": results,
    }


def main() -> int:
    args = _parse_args()
    cases = load_cases(args.cases)
    if not cases:
        print("No cases found.")
        return 1

    report = run_retrieval_eval(cases, k=args.k, mode_choice=args.mode)
    results = report["results"]
    valid_count = int(report.get("valid_count", 0))
    recall_at_k = float(report.get("recall_at_k", 0.0))
    mrr = float(report.get("mrr_at_k", 0.0))

    print("=== RAG RETRIEVAL EVAL ===")
    print(f"cases: {len(results)} | valid: {valid_count} | k: {args.k}")
    print(f"Recall@{args.k}: {recall_at_k:.4f}")
    print(f"MRR@{args.k}: {mrr:.4f}")
    print("\nPer-case:")
    for item in results:
        name = item.get("name", "unknown")
        if item.get("error"):
            print(f"- {name}: ERROR | {item['error']}")
            continue
        print(
            f"- {name}: hit={item.get('hit')} rank={item.get('rank')} rr={float(item.get('rr', 0.0)):.4f} "
            f"mode={item.get('resolved_query_mode')} profile={item.get('resolved_profile')}"
        )

    if args.report.strip():
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[report saved] {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
