from __future__ import annotations

from pathlib import Path

from app import ask_question, process_uploaded_pdfs


TEST_CASES = [
    {
        "pdf": Path("sample_dialog_120_sentences.pdf"),
        "checks": [
            {
                "question": "what is this document about?",
                "must_include_any": ["Speaker A", "Speaker B", "testing", "dataset"],
            },
            {
                "question": "give me an outline of this document",
                "must_include_any": ["Speaker A", "Speaker B", "testing", "dataset"],
            },
        ],
    },
    {
        "pdf": Path("sample_natural_dialog_127.pdf"),
        "checks": [
            {
                "question": "what is this document about?",
                "must_include_any": ["Mia", "Leo", "release", "PDF", "conversation"],
            },
            {
                "question": "From the conversation, what are Mia and Leo likely roles or job functions?",
                "must_include_any": ["Mia", "Leo", "工程師", "QA", "產品", "團隊", "software", "AI"],
            },
            {
                "question": "Based on this conversation, what are the speakers likely working on?",
                "must_include_any": ["軟體", "AI", "PDF", "release", "deploy", "系統"],
            },
            {
                "question": "從這段對話判斷，Mia 和 Leo 比較像什麼身分或職能？",
                "must_include_any": ["工程師", "QA", "產品", "團隊", "Mia", "Leo"],
            },
            {
                "question": "從這段對話看，Mia 和 Leo 的關係比較像什麼？",
                "must_include_any": ["同事", "團隊", "合作", "夥伴"],
            },
            {
                "question": "請整理這段對話的會議重點或工作重點。",
                "must_include_any": ["部署", "測試", "會議", "工作", "協作", "專案"],
            },
        ],
    },
]


def main() -> None:
    failures: list[str] = []

    for case in TEST_CASES:
        pdf = case["pdf"]
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

        history: list[dict[str, str]] = []
        conv_state = list(conversations)
        for check in case["checks"]:
            question = check["question"]
            history, _, _, _, conv_state = ask_question(
                question,
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
            print(f"\nPDF: {pdf.name}\nQ: {question}\nA: {answer[:900]}")

            if "ICD" in answer or "icd" in answer:
                failures.append(f"{pdf.name} 的一般問題不應被誤導成 ICD。")
            if "根據文件內容，我找不到相關資訊" in answer:
                failures.append(f"{pdf.name} 的問題 `{question}` 不應回找不到。")
            if not any(token in answer for token in check["must_include_any"]):
                failures.append(f"{pdf.name} 的問題 `{question}` 缺少預期重點。")

    if failures:
        print("\nGeneral PDF regression FAILED")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    print("\nGeneral PDF regression PASSED")


if __name__ == "__main__":
    main()
