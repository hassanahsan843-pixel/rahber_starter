#!/usr/bin/env python3
"""E2 – Query coverage checker.

For every live query, finds the most similar golden question using
bag-of-words cosine similarity and reports whether it exceeds the
coverage threshold.

Exits with non-zero status when coverage is below the configured threshold.

Usage:
    python eval/coverage.py
    python eval/coverage.py --threshold 0.70 --out eval/results/coverage.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COVERAGE_THRESHOLD = float(os.environ.get("COVERAGE_THRESHOLD", "0.65"))


def _bow_vector(text: str, dim: int = 512) -> list[float]:
    import re
    vec = [0.0] * dim
    for w in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (n1 * n2)))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_coverage(
    golden_path: Path,
    live_path: Path,
    threshold: float = COVERAGE_THRESHOLD,
    out_path: Path | None = None,
) -> dict:
    golden = load_jsonl(golden_path)
    live = load_jsonl(live_path)

    golden_vecs = [(q, _bow_vector(q["question"])) for q in golden]

    covered = 0
    uncovered_rows = []
    all_rows = []

    for lq in live:
        q_text = lq["question"]
        q_vec = _bow_vector(q_text)
        best_sim = 0.0
        best_match = ""
        for gq, gvec in golden_vecs:
            sim = _cosine(q_vec, gvec)
            if sim > best_sim:
                best_sim = sim
                best_match = gq["question"]

        is_covered = best_sim >= threshold
        if is_covered:
            covered += 1
        else:
            uncovered_rows.append({
                "id": lq["id"],
                "question": q_text,
                "best_golden": best_match,
                "similarity": round(best_sim, 4),
                "category": lq.get("category", "unknown"),
            })

        all_rows.append({
            "id": lq["id"],
            "question": q_text,
            "best_golden": best_match,
            "similarity": round(best_sim, 4),
            "covered": is_covered,
            "category": lq.get("category", "unknown"),
        })

    total = len(live)
    coverage_pct = covered / total if total else 0.0

    # Group uncovered by category
    uncovered_by_cat: dict[str, int] = {}
    for row in uncovered_rows:
        cat = row["category"]
        uncovered_by_cat[cat] = uncovered_by_cat.get(cat, 0) + 1

    print(f"\n=== E2: Query Coverage Report ===\n")
    print(f"  Golden questions    : {len(golden)}")
    print(f"  Live queries        : {total}")
    print(f"  Covered (≥{threshold:.0%}) : {covered}")
    print(f"  Uncovered           : {total - covered}")
    print(f"  Coverage            : {coverage_pct:.1%}")
    print(f"  Threshold           : {threshold:.0%}")
    print()
    if uncovered_rows:
        print(f"  Uncovered topics by category:")
        for cat, count in sorted(uncovered_by_cat.items()):
            print(f"    {cat:<25} {count}")
        print()
        print(f"  Sample uncovered queries:")
        for row in uncovered_rows[:5]:
            print(f"    [{row['id']}] sim={row['similarity']:.3f}  '{row['question'][:60]}'")
    print()

    result = {
        "threshold": threshold,
        "golden_count": len(golden),
        "live_count": total,
        "covered": covered,
        "uncovered": total - covered,
        "coverage_pct": round(coverage_pct, 4),
        "passed": coverage_pct >= threshold,
        "uncovered_by_category": uncovered_by_cat,
        "rows": all_rows,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  Coverage report saved to {out_path}")

    status = "PASS" if result["passed"] else "FAIL"
    print(f"  Coverage gate: {status}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="E2: Query coverage checker")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    parser.add_argument("--live", type=Path, default=HERE / "live_queries.jsonl")
    parser.add_argument("--threshold", type=float, default=COVERAGE_THRESHOLD)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run_coverage(args.golden, args.live, args.threshold, args.out)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
