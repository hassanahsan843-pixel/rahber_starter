#!/usr/bin/env python3
"""C2 – Configurable release gate.

Loads baseline results, candidate results, and thresholds from config.
Prints PASS or FAIL per metric. Exits 0 for overall PASS, 1 for FAIL.

Usage:
    python eval/release_gate.py \\
        --baseline eval/results/all_baseline.json \\
        --candidate eval/results/all_candidate.json \\
        --thresholds eval/config/thresholds.yaml \\
        --out eval/results/gate_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _get(d: dict, *keys: str, default: float = 0.0) -> float:
    """Navigate nested dict with dotted key path."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return float(d) if d is not None else default


def run_gate(baseline: dict, candidate: dict, thresholds: dict) -> tuple[list[str], list[str]]:
    """Return (passes, failures) as lists of human-readable messages."""
    passes: list[str] = []
    failures: list[str] = []

    def check(label: str, actual: float, threshold: float,
               direction: str = "min", baseline_val: float | None = None,
               max_drop: float | None = None) -> None:
        if max_drop is not None and baseline_val is not None:
            # Round away float noise: 0.80 - 0.75 is 0.05000000000000004, which
            # would otherwise fail a drop that is exactly at the allowed limit.
            drop = round(baseline_val - actual, 9)
            if drop > max_drop:
                failures.append(f"FAIL  {label}: {actual:.4f} (dropped {drop:.4f} vs baseline {baseline_val:.4f}, max drop {max_drop})")
            else:
                passes.append(f"PASS  {label}: {actual:.4f} (drop {drop:.4f} ≤ {max_drop})")
        elif direction == "min":
            if actual < threshold:
                failures.append(f"FAIL  {label}: {actual:.4f} < threshold {threshold}")
            else:
                passes.append(f"PASS  {label}: {actual:.4f} ≥ threshold {threshold}")
        else:  # max
            if actual > threshold:
                failures.append(f"FAIL  {label}: {actual:.4f} > threshold {threshold}")
            else:
                passes.append(f"PASS  {label}: {actual:.4f} ≤ threshold {threshold}")

    # Extract metrics from result files
    # Candidate results structure: {retrieval: {baseline: {...}}, e2e: {...}}
    cand_ret = _get(candidate, "retrieval", "baseline", "recall_at_5")
    base_ret = _get(baseline, "retrieval", "baseline", "recall_at_5")
    cand_ret3 = _get(candidate, "retrieval", "baseline", "recall_at_3")
    cand_mrr = _get(candidate, "retrieval", "baseline", "mrr")
    cand_ndcg = _get(candidate, "retrieval", "baseline", "ndcg_at_5")
    cand_faith = _get(candidate, "e2e", "faithfulness_mean")
    cand_cit = _get(candidate, "e2e", "citation_validity_mean")
    cand_abst = _get(candidate, "e2e", "abstention_correctness")
    cand_p95 = _get(candidate, "e2e", "latency_p95_s")

    check("Recall@5 drop vs baseline",
          cand_ret, 0.0,
          baseline_val=base_ret,
          max_drop=thresholds.get("recall_at_5_min_drop", 0.05))

    check("Recall@3 absolute floor",
          cand_ret3, thresholds.get("recall_at_3_min", 0.50))

    check("MRR",
          cand_mrr, thresholds.get("mrr_min", 0.40))

    check("nDCG@5",
          cand_ndcg, thresholds.get("ndcg_at_5_min", 0.40))

    check("Faithfulness",
          cand_faith, thresholds.get("faithfulness_min", 0.60))

    check("Citation validity",
          cand_cit, thresholds.get("citation_validity_min", 0.90))

    check("Abstention correctness",
          cand_abst, thresholds.get("abstention_correctness_min", 1.0))

    check("Latency p95",
          cand_p95, thresholds.get("latency_p95_max_s", 10.0), direction="max")

    return passes, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="C2: Release gate")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Baseline evaluation results JSON")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Candidate evaluation results JSON")
    parser.add_argument("--thresholds", type=Path,
                        default=HERE / "config" / "thresholds.yaml")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    thresholds = load_yaml(args.thresholds)

    passes, failures = run_gate(baseline, candidate, thresholds)

    print("\n=== C2: Release Gate ===\n")
    for msg in passes:
        print(f"  {msg}")
    for msg in failures:
        print(f"  {msg}")

    overall = "PASS" if not failures else "FAIL"
    print(f"\n  Overall: {overall} ({len(passes)} passed, {len(failures)} failed)\n")

    report = {
        "overall": overall,
        "passed": len(passes),
        "failed": len(failures),
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "thresholds": str(args.thresholds),
        "details": {"passes": passes, "failures": failures},
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  Gate report saved to {args.out}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
