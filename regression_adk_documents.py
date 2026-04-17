from __future__ import annotations

from pathlib import Path
import time

from adk_apps.pdf_qa_agent import tools


MAX_SECONDS = 90.0


def _require_contains(text: str, expected: list[str], label: str) -> list[str]:
    failures: list[str] = []
    for token in expected:
        if token not in text:
            failures.append(f"{label} missing token: {token}")
    return failures


def main() -> None:
    started_at = time.perf_counter()
    failures: list[str] = []

    pdf_candidates = sorted(Path('.').glob('ICD*.pdf'))
    csv_candidates = sorted(Path('.').glob('icd_csv_test_sample.csv'))
    if not pdf_candidates:
        raise SystemExit('Missing ICD PDF in project root.')
    if not csv_candidates:
        raise SystemExit('Missing icd_csv_test_sample.csv in project root.')

    pdf_path = str(pdf_candidates[0])
    csv_path = str(csv_candidates[0])

    set_pdf = tools.set_active_document(pdf_path)
    failures += _require_contains(set_pdf, [pdf_candidates[0].name], 'set_active_document(pdf)')

    pdf_summary = tools.summarize_document(pdf_path)
    if len(pdf_summary.strip()) < 20:
        failures.append('summarize_document(pdf) output too short')

    pdf_pos = tools.get_document_position(34, pdf_path)
    failures += _require_contains(pdf_pos, ['34'], 'get_document_position(pdf)')

    set_csv = tools.set_active_document(csv_path)
    failures += _require_contains(set_csv, [csv_candidates[0].name], 'set_active_document(csv)')

    csv_row = tools.get_document_position(4, csv_path)
    failures += _require_contains(csv_row, ['code: 6C21'], 'get_document_position(csv)')

    csv_answer = tools.answer_document_question('what is 6C41', csv_path)
    failures += _require_contains(csv_answer, ['6C41'], 'answer_document_question(csv)')

    doc_list = tools.list_available_documents(limit=10)
    failures += _require_contains(doc_list, ['.pdf', '.csv'], 'list_available_documents')

    elapsed = time.perf_counter() - started_at
    print(f'Elapsed: {elapsed:.2f}s')
    if elapsed > MAX_SECONDS:
        failures.append(f'elapsed exceeds {MAX_SECONDS:.0f}s: {elapsed:.2f}s')

    if failures:
        print('Regression FAILED')
        for item in failures:
            print(f'- {item}')
        raise SystemExit(1)

    print('Regression PASSED')


if __name__ == '__main__':
    main()
