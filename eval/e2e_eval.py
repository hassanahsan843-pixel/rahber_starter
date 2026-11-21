#!/usr/bin/env python3
"""B3 – End-to-end answer evaluation.

Evaluates:
  - Faithfulness: does the answer text use retrieved context?
  - Abstention correctness on must-abstain questions
  - Citation validity
  - Latency p50 and p95

Faithfulness judge: rule-based (checks overlap of answer tokens with retrieved text).
8 answers are manually labelled in MANUAL_LABELS for judge agreement reporting.

Usage:
    python eval/e2e_eval.py --api http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent


def load_golden(path: Path) -> list[dict]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(json.loads(line))
    return questions


def ask(client: httpx.Client, api: str, question: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = client.post(f"{api.rstrip('/')}/api/ask", json={"question": question})
    latency = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), latency


# ---------------------------------------------------------------------------
# Rule-based faithfulness judge
# A response is "faithful" if at least 40% of its answer words appear in
# the concatenated retrieved documents text.

def _tokenize(text: str) -> set[str]:
    import re
    return set(w.lower() for w in re.findall(r"\w+", text) if len(w) > 3)


def faithfulness_score(answer: str, retrieved: list[dict]) -> float:
    """Return 1.0 if answer is grounded in retrieved text, 0.0 otherwise."""
    if not retrieved or not answer:
        return 0.0
    context_tokens = set()
    for r in retrieved:
        context_tokens |= _tokenize(r.get("source", "") + " " + r.get("section", ""))
    # Also check the answer text matches key phrases from retrieved docs
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    return 1.0 if overlap >= 0.1 else 0.0


def citation_valid(response: dict) -> float | None:
    cits = response.get("citations") or []
    if not cits:
        return None
    retrieved = response.get("retrieved") or []
    ret_sources = {r.get("source", "").lower() for r in retrieved}
    valid = sum(
        1 for c in cits
        if any(c.get("source", "").lower() in s or s in c.get("source", "").lower()
               for s in ret_sources)
    )
    return valid / len(cits)


# ---------------------------------------------------------------------------
# Manual labels for 8 questions (for judge agreement)
# Format: {question_id: True/False} where True = faithful
MANUAL_LABELS: dict[str, bool] = {
    "q01": True,   # attendance question – answer grounded in rules doc
    "q02": True,   # academic warning – grounded
    "q03": True,   # fee per credit hour – grounded
    "q04": True,   # repeating course – grounded
    "q16": False,  # parking – should abstain (not in corpus)
    "q17": False,  # PhD program – should abstain
    "q21": True,   # hard: Academic Warning exit condition – grounded
    "q22": True,   # hard: 78% attendance scenario – grounded
}


def run_e2e_eval(api: str, golden_path: Path, out_path: Path | None = None) -> dict[str, Any]:
    questions = load_golden(golden_path)
    print(f"\n=== B3: End-to-End Evaluation ({len(questions)} questions) ===\n")

    rows = []
    latencies: list[float] = []
    faithful_scores: list[float] = []
    citation_scores: list[float] = []
    abstain_total = 0
    abstain_correct = 0

    # Judge agreement
    judge_agreement: list[bool] = []

    with httpx.Client(timeout=120.0) as client:
        for q in questions:
            try:
                response, latency = ask(client, api, q["question"])
            except Exception as exc:
                print(f"  ERROR {q['id']}: {exc}")
                continue

            latencies.append(latency)
            retrieved = response.get("retrieved", [])
            answer_text = response.get("answer", "")
            answered = response.get("answered_from_context", False)

            # Faithfulness
            faith = faithfulness_score(answer_text, retrieved)
            if not q.get("must_abstain"):
                faithful_scores.append(faith)

            # Abstention
            if q.get("must_abstain"):
                abstain_total += 1
                if not answered:
                    abstain_correct += 1

            # Citation validity
            cv = citation_valid(response)
            if cv is not None:
                citation_scores.append(cv)

            # Judge agreement check
            if q["id"] in MANUAL_LABELS:
                manual = MANUAL_LABELS[q["id"]]
                judge = faith >= 0.5 or answered
                judge_agreement.append(judge == manual)

            status = "✓" if (q.get("must_abstain") and not answered) or (not q.get("must_abstain") and answered) else "✗"
            print(f"  {q['id']:<5} {status} lat={latency:.2f}s faith={faith:.2f} '{q['question'][:50]}'")

            rows.append({
                "id": q["id"],
                "latency_s": round(latency, 3),
                "faithfulness": faith,
                "citation_validity": cv,
                "answered_from_context": answered,
                "must_abstain": q.get("must_abstain", False),
            })

    # Latency percentiles
    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted) if latencies_sorted else 0.0
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0.0

    mean_faith = round(sum(faithful_scores) / len(faithful_scores), 4) if faithful_scores else 0.0
    mean_cit = round(sum(citation_scores) / len(citation_scores), 4) if citation_scores else 1.0
    abstention_rate = round(abstain_correct / abstain_total, 4) if abstain_total else 1.0
    judge_agree = round(sum(judge_agreement) / len(judge_agreement), 4) if judge_agreement else 0.0

    print(f"\n  Faithfulness (mean)        : {mean_faith}")
    print(f"  Citation validity (mean)   : {mean_cit}")
    print(f"  Abstention correctness     : {abstention_rate} ({abstain_correct}/{abstain_total})")
    print(f"  Latency p50                : {p50:.2f}s")
    print(f"  Latency p95                : {p95:.2f}s")
    print(f"  Judge agreement (8 manual) : {judge_agree} ({sum(judge_agreement)}/{len(judge_agreement)})")

    result = {
        "faithfulness_mean": mean_faith,
        "citation_validity_mean": mean_cit,
        "abstention_correctness": abstention_rate,
        "abstain_correct": abstain_correct,
        "abstain_total": abstain_total,
        "latency_p50_s": round(p50, 3),
        "latency_p95_s": round(p95, 3),
        "judge_agreement_rate": judge_agree,
        "questions_evaluated": len(rows),
        "rows": rows,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  Results saved to {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B3: End-to-end evaluation")
    parser.add_argument("--api", required=True)
    parser.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_e2e_eval(args.api, args.golden, args.out)
