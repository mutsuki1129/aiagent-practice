from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from adk_apps.pdf_qa_agent import tools


DEFAULT_EVAL_FILES = [
    Path("adk_apps/pdf_qa_agent/evals/general_pdf_eval.json"),
    Path("adk_apps/pdf_qa_agent/evals/csv_document_eval.json"),
]


def _load_specs(eval_file: Path) -> list[dict[str, Any]]:
    data = json.loads(eval_file.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{eval_file} must be a JSON array.")
    return [item for item in data if isinstance(item, dict)]


def _invoke_spec(spec: dict[str, Any]) -> str:
    tool_name = str(spec.get("tool", "answer_document_question")).strip()
    file_path = str(spec.get("file_path") or spec.get("pdf_path") or "").strip() or None

    if tool_name == "get_document_position":
        position_number = int(spec.get("position_number", 0) or 0)
        if position_number <= 0:
            raise ValueError(f"{spec.get('name', '<unknown>')} missing valid position_number.")
        return tools.get_document_position(position_number=position_number, file_path=file_path)

    question = str(spec.get("question", "")).strip()
    if not question:
        raise ValueError(f"{spec.get('name', '<unknown>')} missing question.")
    return tools.answer_document_question(question=question, file_path=file_path)


def _evaluate_specs(eval_file: Path) -> list[str]:
    failures: list[str] = []
    specs = _load_specs(eval_file)
    for spec in specs:
        name = str(spec.get("name", "<unnamed>"))
        expected = [str(x) for x in spec.get("expected_keywords", [])]
        output = _invoke_spec(spec)
        for token in expected:
            if token not in output:
                failures.append(f"{eval_file.name}/{name} missing keyword: {token}")
    return failures


def main() -> None:
    started = time.perf_counter()
    failures: list[str] = []

    for eval_file in DEFAULT_EVAL_FILES:
        if not eval_file.exists():
            print(f"[SKIP] {eval_file} not found.")
            continue
        failures.extend(_evaluate_specs(eval_file))

    elapsed = time.perf_counter() - started
    print(f"Elapsed: {elapsed:.2f}s")
    if failures:
        print("ADK Tool Eval FAILED")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    print("ADK Tool Eval PASSED")


if __name__ == "__main__":
    main()

