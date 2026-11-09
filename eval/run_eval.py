#!/usr/bin/env python3
"""Score a running RAHBAR against a golden set, and gate CI on the result (section 5).

    python eval/run_eval.py --api http://<ingress>                 # your corpus
    python eval/run_eval.py --api ... --golden eval/fixtures/golden.fixture.yaml
    python eval/run_eval.py --api ... --update-baseline            # after a deliberate improvement

Exit code 0 when every gate passes, 1 when any fails. CI relies on that exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).resolve().parent

# The gate from section 5.4. Change these only with a reason written in docs/PIPELINE.md.
MAX_RECALL5_DROP = 0.05        # recall@5 may fall at most 5 points below the baseline
MIN_CITATION_VALIDITY = 0.95   # at least 95% of citations must match retrieved passages
REQUIRED_ABSTENTION = 1.0      # every must-abstain question must be declined


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []
    if not questions:
        sys.exit(f"no questions in {path}")
    return questions


def ask(client: httpx.Client, api: str, question: str) -> dict:
    """POST one question and return the AskResponse as a dict (see api/app/schemas.py)."""
    response = client.post(f"{api.rstrip('/')}/api/ask", json={"question": question})
    response.raise_for_status()
    return response.json()


def _match_passage(rel: dict, ret: dict) -> bool:
    """Check if retrieved passage matches golden relevant item by source and section."""
    rel_src = str(rel.get("source", "")).lower()
    ret_src = str(ret.get("source", "")).lower()
    rel_sec = str(rel.get("section", "")).lower()
    ret_sec = str(ret.get("section", "")).lower()

    src_match = (rel_src in ret_src) or (ret_src in rel_src)
    # Section match allows substring or clause number matching (e.g., '1.21' in '1.21 Attendance')
    sec_match = (rel_sec in ret_sec) or (ret_sec in rel_sec)
    return src_match and sec_match


def recall_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> float | None:
    """Fraction of the relevant (source, section) pairs found in the top k retrieved.
    Return None when the question has no relevant passages (must-abstain questions).
    """
    if not relevant:
        return None
    top_k_retrieved = retrieved[:k]
    hits = 0
    for rel in relevant:
        if any(_match_passage(rel, ret) for ret in top_k_retrieved):
            hits += 1
    return hits / len(relevant)


def citation_validity(response: dict) -> float | None:
    """Fraction of the response's citations that match a passage in response['retrieved'].
    Return None when there are no citations.
    """
    citations = response.get("citations") or []
    if not citations:
        return None
    retrieved = response.get("retrieved") or []
    valid = 0
    for cit in citations:
        if any(_match_passage(cit, ret) for ret in retrieved):
            valid += 1
    return valid / len(citations)


def summarise(rows: list[dict]) -> dict:
    """Overall recall@3, recall@5, citation validity and abstention correctness, plus
    recall@5 broken down by question type. This dict becomes results.json.
    """
    recall3_scores: list[float] = []
    recall5_scores: list[float] = []
    by_type: dict[str, list[float]] = {}
    citation_scores: list[float] = []
    abstain_total = 0
    abstain_correct = 0

    for row in rows:
        q_type = row.get("type", "ordinary")
        relevant = row.get("relevant") or []
        resp = row.get("response") or {}
        retrieved = resp.get("retrieved") or []

        if row.get("must_abstain"):
            abstain_total += 1
            if not resp.get("answered_from_context", True):
                abstain_correct += 1
        else:
            r3 = recall_at_k(relevant, retrieved, k=3)
            r5 = recall_at_k(relevant, retrieved, k=5)
            if r3 is not None:
                recall3_scores.append(r3)
            if r5 is not None:
                recall5_scores.append(r5)
                by_type.setdefault(q_type, []).append(r5)

        cv = citation_validity(resp)
        if cv is not None:
            citation_scores.append(cv)

    avg_recall3 = sum(recall3_scores) / len(recall3_scores) if recall3_scores else 0.0
    avg_recall5 = sum(recall5_scores) / len(recall5_scores) if recall5_scores else 0.0
    avg_citation = sum(citation_scores) / len(citation_scores) if citation_scores else 1.0
    abstention_rate = (abstain_correct / abstain_total) if abstain_total > 0 else 1.0

    by_type_avg = {
        t: round(sum(scores) / len(scores), 4) for t, scores in by_type.items() if scores
    }

    return {
        "recall_at_3": round(avg_recall3, 4),
        "recall_at_5": round(avg_recall5, 4),
        "by_type": by_type_avg,
        "citation_validity": round(avg_citation, 4),
        "abstention_correctness": round(abstention_rate, 4),
    }


def gate(results: dict, baseline: dict | None) -> list[str]:
    """Return one message per failed gate (section 5.4); an empty list means pass."""
    failures: list[str] = []

    # 1. Recall drop gate
    if baseline and "recall_at_5" in baseline:
        drop = baseline["recall_at_5"] - results.get("recall_at_5", 0.0)
        if drop > MAX_RECALL5_DROP:
            failures.append(
                f"recall@5 dropped by {drop:.3f} below baseline (max allowed drop is {MAX_RECALL5_DROP})"
            )

    # 2. Honest abstention gate (must be 100%)
    abstention_rate = results.get("abstention_correctness", 1.0)
    if abstention_rate < REQUIRED_ABSTENTION:
        failures.append(
            f"abstention correctness is {abstention_rate:.1%} (required 100%)"
        )

    # 3. Citation validity gate (must be >= 95%)
    cit_validity = results.get("citation_validity", 1.0)
    if cit_validity < MIN_CITATION_VALIDITY:
        failures.append(
            f"citation validity fell to {cit_validity:.1%} (minimum threshold is {MIN_CITATION_VALIDITY:.0%})"
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAHBAR against a golden set.")
    parser.add_argument("--api", required=True, help="base URL, e.g. http://192.168.49.2")
    parser.add_argument("--host", help="Host header, when your Ingress rule names a host")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.yaml")
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--update-baseline", action="store_true",
                        help="write these results as the new baseline")
    args = parser.parse_args()

    questions = load_golden(args.golden)
    rows = []
    headers = {"Host": args.host} if args.host else {}
    with httpx.Client(timeout=120.0, headers=headers) as client:
        for q in questions:
            started = time.perf_counter()
            response = ask(client, args.api, q["question"])
            rows.append({
                "id": q["id"], "type": q.get("type", "ordinary"),
                "must_abstain": bool(q.get("must_abstain", False)),
                "relevant": q.get("relevant") or [],
                "expect": q.get("expect"),
                "response": response,
                "seconds": round(time.perf_counter() - started, 2),
            })
            print(f"  {q['id']:<5} {rows[-1]['seconds']:>6.2f}s  {q['question'][:60]}")

    results = summarise(rows)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline.exists() else None
    failures = gate(results, baseline)

    results["golden"] = str(args.golden)
    results["questions"] = len(rows)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {args.out}")
    if args.update_baseline:
        args.baseline.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"baseline updated: {args.baseline}")

    if failures:
        print("\nGATE FAILED")
        for message in failures:
            print(f"  - {message}")
        sys.exit(1)
    print("\ngate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
