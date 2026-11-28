#!/usr/bin/env python3
"""B2 – Component-level retrieval evaluation.

Calculates over the golden set:
  - Recall@3
  - Recall@5
  - MRR (Mean Reciprocal Rank)
  - nDCG@5

Also runs a noise test:
  - Introduces typos into queries
  - Re-runs retrieval and reports metric changes

Usage:
    python eval/retrieval_eval.py --api http://127.0.0.1:8000
    python eval/retrieval_eval.py --api http://127.0.0.1:8000 --out eval/results/retrieval.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import httpx
import yaml

HERE = Path(__file__).resolve().parent


def _normalize(q: dict) -> dict:
    """One schema for both golden formats.

    golden.jsonl: ground_truth_doc + must_abstain.
    golden.yaml (A02 real-corpus set): relevant: [{source, section}] + must_abstain/type.
    Without this, every golden.yaml question had no ground_truth_doc and was
    silently excluded from Recall/MRR/nDCG.
    """
    q = dict(q)
    sources = [str(r.get("source")) for r in (q.get("relevant") or []) if r.get("source")]
    if q.get("ground_truth_doc"):
        sources = [str(q["ground_truth_doc"])] + [s for s in sources if s != q["ground_truth_doc"]]
    q["relevant_sources"] = sources
    q["ground_truth_doc"] = q.get("ground_truth_doc") or (sources[0] if sources else None)
    q["must_abstain"] = bool(q.get("must_abstain")) or q.get("type") in ("unanswerable", "abstain")
    return q


def load_golden(path: Path) -> list[dict]:
    """Load golden.jsonl or golden.yaml, return normalized question dicts."""
    if path.suffix == ".jsonl":
        questions = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        questions = data.get("questions") or []
    return [_normalize(q) for q in questions]


def ask(client: httpx.Client, api: str, question: str) -> dict:
    r = client.post(f"{api.rstrip('/')}/api/ask", json={"question": question})
    r.raise_for_status()
    return r.json()


def _match(rel: dict, ret: dict) -> bool:
    rel_src = str(rel.get("ground_truth_doc", rel.get("source", ""))).lower().strip()
    ret_src = str(ret.get("source", "")).lower().strip()

    if not rel_src or not ret_src:
        return False

    return rel_src in ret_src or ret_src in rel_src


def _is_relevant(q: dict, ret: dict) -> bool:
    sources = q.get("relevant_sources") or [q.get("ground_truth_doc")]
    return any(_match({"ground_truth_doc": s}, ret) for s in sources if s)


def recall_at_k(q: dict, retrieved: list[dict], k: int) -> float | None:
    if q.get("must_abstain") or not q.get("ground_truth_doc"):
        return None
    hits = any(_is_relevant(q, r) for r in retrieved[:k])
    return 1.0 if hits else 0.0


def mrr(q: dict, retrieved: list[dict]) -> float | None:
    if q.get("must_abstain") or not q.get("ground_truth_doc"):
        return None
    for rank, r in enumerate(retrieved, start=1):
        if _is_relevant(q, r):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(q: dict, retrieved: list[dict], k: int) -> float | None:
    if q.get("must_abstain") or not q.get("ground_truth_doc"):
        return None

    for i, r in enumerate(retrieved[:k], start=1):
        if _is_relevant(q, r):
            dcg = 1.0 / math.log2(i + 1)
            idcg = 1.0
            return dcg / idcg

    return 0.0


def _add_typos(text: str, seed: int = 42) -> str:
    """Add simple character-swap typos to 15% of words."""
    rng = random.Random(seed)
    words = text.split()
    result = []
    for w in words:
        if len(w) > 3 and rng.random() < 0.15:
            i = rng.randint(0, len(w) - 2)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        result.append(w)
    return " ".join(result)


def evaluate(questions: list[dict], client: httpx.Client, api: str,
             noise: bool = False, seed: int = 42) -> dict[str, Any]:
    r3s, r5s, mrrs, ndcgs = [], [], [], []
    rows = []

    for q in questions:
        question = q["question"]
        if noise:
            question = _add_typos(question, seed=seed)

        try:
            response = ask(client, api, question)
        except Exception as exc:
            print(f"  ERROR on {q['id']}: {exc}")
            continue

        retrieved = response.get("retrieved", [])

        r3 = recall_at_k(q, retrieved, 3)
        r5 = recall_at_k(q, retrieved, 5)
        m = mrr(q, retrieved)
        n = ndcg_at_k(q, retrieved, 5)

        if r3 is not None:
            r3s.append(r3)
        if r5 is not None:
            r5s.append(r5)
        if m is not None:
            mrrs.append(m)
        if n is not None:
            ndcgs.append(n)

        rows.append({
            "id": q["id"],
            "type": q.get("type"),
            "noise": noise,
            "recall_at_3": r3,
            "recall_at_5": r5,
            "mrr": m,
            "ndcg_at_5": n,
        })
        def _fmt(v: float | None) -> str:
            # 0.0 is a miss and must be visible; '-' only means "not scored"
            return "-" if v is None else f"{v:.3f}"

        print(f"  {q['id']:<5} R@3={_fmt(r3):>5}  R@5={_fmt(r5):>5}  "
              f"MRR={_fmt(m):>5}  nDCG={_fmt(n):>5}  '{question[:45]}'")

    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "noise": noise,
        "recall_at_3": _avg(r3s),
        "recall_at_5": _avg(r5s),
        "mrr": _avg(mrrs),
        "ndcg_at_5": _avg(ndcgs),
        "questions_evaluated": len(rows),
        "rows": rows,
    }


def run_retrieval_eval(api: str, golden_path: Path, out_path: Path | None = None,
                       seed: int = 42, headers: dict[str, str] | None = None) -> dict[str, Any]:
    questions = load_golden(golden_path)
    print(f"\n=== B2: Retrieval Evaluation ({len(questions)} questions) ===")

    with httpx.Client(timeout=60.0, headers=headers) as client:
        print("\n--- Baseline (clean queries) ---")
        baseline = evaluate(questions, client, api, noise=False)

        print("\n--- Noise test (typos in queries) ---")
        noisy = evaluate(questions, client, api, noise=True, seed=seed)

    print(f"\n  Baseline  R@3={baseline['recall_at_3']}  R@5={baseline['recall_at_5']}  "
          f"MRR={baseline['mrr']}  nDCG@5={baseline['ndcg_at_5']}")
    print(f"  Noisy     R@3={noisy['recall_at_3']}  R@5={noisy['recall_at_5']}  "
          f"MRR={noisy['mrr']}  nDCG@5={noisy['ndcg_at_5']}")

    result = {"baseline": baseline, "noisy": noisy}

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  Results saved to {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B2: Retrieval evaluation")
    parser.add_argument("--api", required=True)
    parser.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_retrieval_eval(args.api, args.golden, args.out, args.seed)
