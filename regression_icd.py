from __future__ import annotations

from pathlib import Path
import time

from app import ask_question, process_uploaded_pdfs


MAX_SECONDS = 60.0

EXPECTATIONS = [
    {
        "question": "6B85 是什麼？",
        "must_include": ["反芻-逆流障礙", "6B85", "餵食或進食障礙"],
        "must_exclude": ["排泄障礙"],
    },
    {
        "question": "物質使用所致障礙有哪些？",
        "must_include": ["6C40", "6C49", "物質使用所致障礙"],
        "must_exclude": [],
    },
    {
        "question": "請整理這份 ICD-11 文件的大綱。",
        "must_include": ["ICD-11", "精神", "神經發育障礙"],
        "must_exclude": ["找不到相關資訊", "additional_kwargs"],
    },
    {
        "question": "第 34 頁有什麼內容？",
        "must_include": ["第 34 頁", "6C41", "大麻素類物質使用所致障礙"],
        "must_exclude": [],
    },
    {
        "question": "軀體不適或身體體驗障礙有哪些？",
        "must_include": ["6C20", "6C21", "軀體不適"],
        "must_exclude": [],
    },
    {
        "question": "衝動控制障礙有哪些？",
        "must_include": ["6C70", "6C71", "6C72", "6C73"],
        "must_exclude": [],
    },
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

    failures: list[str] = []
    history: list[dict[str, str]] = []
    conv_state = list(conversations)

    for spec in EXPECTATIONS:
        if time.perf_counter() - started_at > MAX_SECONDS:
            failures.append(f"測試超時，超過 {MAX_SECONDS:.0f} 秒。")
            break

        history, _, _, _, conv_state = ask_question(
            spec["question"],
            history,
            retriever,
            document_index,
            "Gemma 4 (本地)",
            "通用推理模式",
            all_docs,
            conv_state,
            detected_mode,
        )
        answer = history[-1]["content"] if history else ""
        print(f"\nQ: {spec['question']}\nA: {answer[:800]}")

        for token in spec["must_include"]:
            if token not in answer:
                failures.append(f"問題 `{spec['question']}` 缺少關鍵字：{token}")

        for token in spec["must_exclude"]:
            if token in answer:
                failures.append(f"問題 `{spec['question']}` 不應包含：{token}")

    elapsed = time.perf_counter() - started_at
    print(f"\nElapsed: {elapsed:.1f}s")

    if failures:
        print("\nRegression FAILED")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    print("\nRegression PASSED")


if __name__ == "__main__":
    main()
