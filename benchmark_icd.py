from __future__ import annotations

from pathlib import Path
import time

from app import ask_question, process_uploaded_pdfs


QUESTIONS = [
    "這份文件主要在講什麼？",
    "請整理這份 ICD-11 文件的大綱。",
    "神經發育障礙有哪些分類？",
    "智力發展障礙是什麼？",
    "注意力不足過動症在哪一類？",
    "6A05 是什麼？",
    "物質使用所致障礙有哪些？",
    "物質使用障礙有哪些相關分類？",
    "急性短暫性精神病性障礙的代碼是什麼？",
    "6B85 是什麼？",
    "第 1 頁有什麼內容？",
    "第 34 頁有什麼內容？",
]


def main() -> None:
    started_at = time.perf_counter()
    pdf = next(Path(".").glob("ICD*.pdf"))
    (
        retriever,
        document_index,
        status,
        _,
        _,
        _,
        _,
        _,
        all_docs,
        conversations,
        detected_mode,
    ) = process_uploaded_pdfs([str(pdf)])
    print(status)

    for model_choice in ("Gemma 4 (本地)", "Gemini (雲端)"):
        print(f"\n=== {model_choice} ===")
        history: list[dict[str, str]] = []
        conv_state = list(conversations)
        model_started_at = time.perf_counter()

        for question in QUESTIONS:
            question_started_at = time.perf_counter()
            history, _, _, preview, conv_state = ask_question(
                question,
                history,
                retriever,
                document_index,
                model_choice,
                "通用推理模式",
                all_docs,
                conv_state,
                detected_mode,
            )
            answer = history[-1]["content"] if history else ""
            elapsed = time.perf_counter() - question_started_at
            print(f"\nQ: {question}")
            print(f"A: {answer[:500]}")
            print(f"P: {preview[:180]}")
            print(f"T: {elapsed:.2f}s")

        model_elapsed = time.perf_counter() - model_started_at
        print(f"\n{model_choice} elapsed: {model_elapsed:.1f}s")

    elapsed = time.perf_counter() - started_at
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
