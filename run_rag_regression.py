from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="One-click local wrapper for rag_retrieval_regression.py")
    parser.add_argument("--cases", default="docs/retrieval_eval_cases.example.json")
    parser.add_argument("--baseline", default="reports/rag_retrieval_baseline.json")
    parser.add_argument("--report", default="reports/rag_retrieval_latest.json")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--allow-missing-baseline", action="store_true")
    parser.add_argument("--max-recall-drop", type=float, default=0.01)
    parser.add_argument("--max-mrr-drop", type=float, default=0.01)
    parser.add_argument("--max-error-cases", type=int, default=0)
    parser.add_argument("--mode", default="通用推理模式")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "rag_retrieval_regression.py",
        "--cases",
        args.cases,
        "--baseline",
        args.baseline,
        "--report",
        args.report,
        "--k",
        str(args.k),
        "--max-recall-drop",
        str(args.max_recall_drop),
        "--max-mrr-drop",
        str(args.max_mrr_drop),
        "--max-error-cases",
        str(args.max_error_cases),
        "--mode",
        args.mode,
    ]
    if args.update_baseline:
        cmd.append("--update-baseline")
    if args.allow_missing_baseline:
        cmd.append("--allow-missing-baseline")

    completed = subprocess.run(cmd)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
