#!/usr/bin/env python3
"""B4 – Stochasticity evaluation.

Runs the retrieval + E2E evaluation twice with the same seed and reports
the difference between runs. Any variation indicates non-determinism.

Usage:
    python eval/stochasticity_eval.py --api http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_stochasticity_eval(api: str, golden_path: Path, out_path: Path | None = None) -> dict:
    from eval.e2e_eval import run_e2e_eval
    from eval.retrieval_eval import run_retrieval_eval

    print("\n=== B4: Stochasticity Evaluation ===")
    print("Running evaluation twice with the same configuration...\n")

    # Run 1
    print("--- Run 1 ---")
    ret1 = run_retrieval_eval(api, golden_path, seed=42)
    e2e1 = run_e2e_eval(api, golden_path)

    # Run 2
    print("\n--- Run 2 ---")
    ret2 = run_retrieval_eval(api, golden_path, seed=42)
    e2e2 = run_e2e_eval(api, golden_path)

    # Compare
    metrics = ["recall_at_3", "recall_at_5", "mrr", "ndcg_at_5"]
    print("\n--- Retrieval metric differences (Run1 - Run2) ---")
    ret_diff = {}
    for m in metrics:
        d = abs(ret1["baseline"].get(m, 0) - ret2["baseline"].get(m, 0))
        ret_diff[m] = round(d, 6)
        print(f"  {m:<20} diff={d:.6f}")

    e2e_metrics = ["faithfulness_mean", "citation_validity_mean",
                   "abstention_correctness", "latency_p50_s", "latency_p95_s"]
    print("\n--- E2E metric differences (Run1 - Run2) ---")
    e2e_diff = {}
    for m in e2e_metrics:
        d = abs(e2e1.get(m, 0) - e2e2.get(m, 0))
        e2e_diff[m] = round(d, 6)
        print(f"  {m:<30} diff={d:.6f}")

    result = {
        "run1_retrieval": ret1["baseline"],
        "run2_retrieval": ret2["baseline"],
        "retrieval_diffs": ret_diff,
        "run1_e2e": {m: e2e1.get(m) for m in e2e_metrics},
        "run2_e2e": {m: e2e2.get(m) for m in e2e_metrics},
        "e2e_diffs": e2e_diff,
        "explanation": (
            "Retrieval differences come from online LLM variability (temperature>0). "
            "Offline/deterministic fallback embedding produces zero retrieval diff. "
            "Latency differences are due to network jitter."
        ),
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  Results saved to {out_path}")

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE.parent))

    parser = argparse.ArgumentParser(description="B4: Stochasticity evaluation")
    parser.add_argument("--api", required=True)
    parser.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_stochasticity_eval(args.api, args.golden, args.out)
