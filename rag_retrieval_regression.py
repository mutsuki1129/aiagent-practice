from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_retrieval_eval import load_cases, run_retrieval_eval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command RAG retrieval regression with baseline comparison.")
    parser.add_argument("--cases", required=True, help="Eval cases JSON path")
    parser.add_argument("--baseline", default="reports/rag_retrieval_baseline.json", help="Baseline JSON path")
    parser.add_argument("--report", default="reports/rag_retrieval_latest.json", help="Latest report JSON path")
    parser.add_argument("--k", type=int, default=8, help="Top-k")
    parser.add_argument("--mode", default="通用推理模式", help="Mode choice label")
    parser.add_argument(
        "--max-recall-drop",
        type=float,
        default=0.01,
        help="Allowed Recall@k drop vs baseline before failing (absolute)",
    )
    parser.add_argument(
        "--max-mrr-drop",
        type=float,
        default=0.01,
        help="Allowed MRR@k drop vs baseline before failing (absolute)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write current report as new baseline and exit success",
    )
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="If baseline is missing, do not fail (still write latest report)",
    )
    parser.add_argument(
        "--max-error-cases",
        type=int,
        default=0,
        help="Allowed number of eval error cases before failing",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _compare(current: dict, baseline: dict) -> dict:
    cur_recall = float(current.get("recall_at_k", 0.0))
    cur_mrr = float(current.get("mrr_at_k", 0.0))
    base_recall = float(baseline.get("recall_at_k", 0.0))
    base_mrr = float(baseline.get("mrr_at_k", 0.0))
    return {
        "current_recall": cur_recall,
        "current_mrr": cur_mrr,
        "baseline_recall": base_recall,
        "baseline_mrr": base_mrr,
        "delta_recall": cur_recall - base_recall,
        "delta_mrr": cur_mrr - base_mrr,
    }


def main() -> int:
    args = _parse_args()
    cases = load_cases(args.cases)
    if not cases:
        print("No eval cases found.")
        return 1

    report = run_retrieval_eval(cases, k=args.k, mode_choice=args.mode)
    report["mode"] = args.mode
    report["cases_path"] = args.cases
    error_count = int(report.get("case_count", 0)) - int(report.get("valid_count", 0))
    report["error_count"] = error_count

    report_path = Path(args.report).expanduser()
    _write_json(report_path, report)
    print(f"[latest report] {report_path}")
    print(f"Recall@{args.k}: {float(report.get('recall_at_k', 0.0)):.4f}")
    print(f"MRR@{args.k}: {float(report.get('mrr_at_k', 0.0)):.4f}")
    print(f"Errors: {error_count}")

    if error_count > max(0, int(args.max_error_cases)):
        print(
            f"\n[REGRESSION DETECTED] error cases {error_count} exceed allowed {max(0, int(args.max_error_cases))}"
        )
        return 4

    baseline_path = Path(args.baseline).expanduser()
    if args.update_baseline:
        _write_json(baseline_path, report)
        print(f"[baseline updated] {baseline_path}")
        return 0

    if not baseline_path.exists():
        print(f"[baseline missing] {baseline_path}")
        if args.allow_missing_baseline:
            return 0
        print("Use --update-baseline to initialize, or --allow-missing-baseline to bypass.")
        return 2

    baseline = _read_json(baseline_path)
    baseline_k = int(baseline.get("k", args.k))
    if baseline_k != int(args.k):
        print(f"[baseline mismatch] baseline k={baseline_k}, current k={int(args.k)}")
        print("Please align k or reinitialize baseline with --update-baseline.")
        return 5

    cmp = _compare(report, baseline)

    print("\n=== Baseline Comparison ===")
    print(f"baseline Recall@{args.k}: {cmp['baseline_recall']:.4f}")
    print(f"baseline MRR@{args.k}: {cmp['baseline_mrr']:.4f}")
    print(f"delta Recall: {cmp['delta_recall']:+.4f}")
    print(f"delta MRR: {cmp['delta_mrr']:+.4f}")

    recall_failed = cmp["delta_recall"] < -abs(args.max_recall_drop)
    mrr_failed = cmp["delta_mrr"] < -abs(args.max_mrr_drop)
    if recall_failed or mrr_failed:
        print("\n[REGRESSION DETECTED]")
        if recall_failed:
            print(
                f"Recall drop {cmp['delta_recall']:+.4f} is below allowed {-abs(args.max_recall_drop):.4f}"
            )
        if mrr_failed:
            print(
                f"MRR drop {cmp['delta_mrr']:+.4f} is below allowed {-abs(args.max_mrr_drop):.4f}"
            )
        return 3

    print("\n[OK] No significant regression detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
