#!/usr/bin/env python3
"""E2 – Query coverage checker.

For every live query, finds the most similar golden question and reports
whether the similarity reaches the coverage threshold.

Similarity method (--method / COVERAGE_METHOD):
  embedding (default)  cosine of EMBED_MODEL vectors from Ollama (the same
                       nomic-embed-text pin that retrieval and B1 use). It
                       recognises paraphrases: "credit hours to graduate" vs
                       "minimum credit hours to complete the BS degree".
  bow                  hashed bag-of-words cosine (the original method). Kept
                       for offline use; it only rewards shared wording.

The cutoff stays at COVERAGE_THRESHOLD (0.65). For nomic-embed-text it sits just
above the midpoint of B1's measurements (paraphrase pairs 0.83, unrelated 0.43).
If the embedding service is unreachable the checker fails (exit 2); it never
silently falls back to bag-of-words.

Exits with non-zero status when coverage is below the configured threshold.

Usage:
    python eval/coverage.py
    python eval/coverage.py --method bow
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
COVERAGE_METHOD = os.environ.get("COVERAGE_METHOD", "embedding")


class EmbeddingUnavailable(RuntimeError):
    """The embedding service could not produce vectors."""


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Vectors from the pinned embedding model (one batched Ollama call)."""
    import httpx

    url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    try:
        res = httpx.post(f"{url}/api/embed", json={"model": model, "input": texts}, timeout=300.0)
        res.raise_for_status()
        vectors = res.json().get("embeddings") or []
    except Exception as exc:
        raise EmbeddingUnavailable(f"{model} at {url}: {exc}") from exc
    if len(vectors) != len(texts) or any(not v for v in vectors):
        raise EmbeddingUnavailable(f"{model} at {url} returned an incomplete batch")
    return vectors


def _vectors(texts: list[str], method: str) -> list[list[float]]:
    if method == "bow":
        return [_bow_vector(t) for t in texts]
    if method == "embedding":
        return _embed_texts(texts) if texts else []
    raise ValueError(f"unknown coverage method {method!r}; expected embedding or bow")


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


def load_jsonl_checked(path: Path) -> tuple[list[dict], list[str]]:
    """Rows with a non-empty "question", plus a message for every line that is not one.

    Nothing is dropped silently: each skipped line is reported with its line number.
    """
    rows: list[dict] = []
    problems: list[str] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}:{n}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(row, dict) or not str(row.get("question") or "").strip():
            problems.append(f"{path.name}:{n}: missing or empty 'question'")
            continue
        row.setdefault("id", f"line{n}")
        rows.append(row)
    return rows, problems


def load_jsonl(path: Path) -> list[dict]:
    return load_jsonl_checked(path)[0]


def run_coverage(
    golden_path: Path,
    live_path: Path,
    threshold: float = COVERAGE_THRESHOLD,
    out_path: Path | None = None,
    similarity_threshold: float | None = None,
    method: str = COVERAGE_METHOD,
) -> dict:
    """`threshold` is the required share of covered live queries. A live query counts
    as covered when its best golden match reaches `similarity_threshold` (defaults to
    the same value, which is how this checker has always been configured)."""
    sim_threshold = threshold if similarity_threshold is None else similarity_threshold
    golden, golden_problems = load_jsonl_checked(golden_path)
    live, live_problems = load_jsonl_checked(live_path)
    ids = [str(lq["id"]) for lq in live]
    duplicate_ids = sorted({i for i in ids if ids.count(i) > 1})

    vectors = _vectors([q["question"] for q in golden] + [q["question"] for q in live], method)
    golden_vecs = list(zip(golden, vectors[:len(golden)]))
    live_vecs = vectors[len(golden):]

    covered = 0
    uncovered_rows = []
    all_rows = []

    for lq, q_vec in zip(live, live_vecs):
        q_text = lq["question"]
        best_sim = 0.0
        best_match = ""
        best_topic = "unknown"
        for gq, gvec in golden_vecs:
            sim = _cosine(q_vec, gvec)
            if sim > best_sim:
                best_sim = sim
                best_match = gq["question"]
                best_topic = str(gq.get("category") or "unknown")

        is_covered = best_sim >= sim_threshold
        if is_covered:
            covered += 1
        else:
            uncovered_rows.append({
                "id": lq["id"],
                "question": q_text,
                "best_golden": best_match,
                "similarity": round(best_sim, 4),
                "category": lq.get("category", "unknown"),
                "nearest_topic": best_topic,
            })

        all_rows.append({
            "id": lq["id"],
            "question": q_text,
            "best_golden": best_match,
            "similarity": round(best_sim, 4),
            "covered": is_covered,
            "category": lq.get("category", "unknown"),
            "nearest_topic": best_topic,
        })

    # Unreadable live queries count against coverage instead of disappearing.
    total = len(live) + len(live_problems)
    coverage_pct = covered / total if total else 0.0

    # Group uncovered by category
    uncovered_by_cat: dict[str, int] = {}
    for row in uncovered_rows:
        cat = row["category"]
        uncovered_by_cat[cat] = uncovered_by_cat.get(cat, 0) + 1
    # Group uncovered queries into topics: the golden category of their nearest question
    uncovered_by_topic: dict[str, list[str]] = {}
    for row in uncovered_rows:
        uncovered_by_topic.setdefault(row["nearest_topic"], []).append(row["question"])

    print("\n=== E2: Query Coverage Report ===\n")
    print(f"  Similarity method   : {method}"
          + (f" ({os.environ.get('EMBED_MODEL', 'nomic-embed-text')})" if method == "embedding" else ""))
    print(f"  Golden questions    : {len(golden)}")
    print(f"  Live queries        : {total}")
    print(f"  Covered (≥{threshold:.0%}) : {covered}")
    print(f"  Uncovered           : {total - covered}")
    print(f"  Coverage            : {coverage_pct:.1%}")
    print(f"  Threshold           : {threshold:.0%} of queries (match similarity >= {sim_threshold:.2f})")
    for msg in golden_problems + live_problems:
        print(f"  WARNING skipped line: {msg}")
    if duplicate_ids:
        print(f"  WARNING duplicate live query ids: {', '.join(duplicate_ids)}")
    print()
    if uncovered_rows:
        print("  Uncovered topics by category:")
        for cat, count in sorted(uncovered_by_cat.items()):
            print(f"    {cat:<25} {count}")
        print()
        print("  Uncovered queries grouped by nearest golden topic:")
        for topic, qs in sorted(uncovered_by_topic.items()):
            print(f"    {topic} ({len(qs)})")
            for q in qs:
                print(f"      - {q[:70]}")
        print()
        print("  Sample uncovered queries:")
        for row in uncovered_rows[:5]:
            print(f"    [{row['id']}] sim={row['similarity']:.3f}  '{row['question'][:60]}'")
    print()

    result = {
        "method": method,
        "embed_model": os.environ.get("EMBED_MODEL", "nomic-embed-text") if method == "embedding" else None,
        "threshold": threshold,
        "golden_count": len(golden),
        "live_count": total,
        "covered": covered,
        "uncovered": total - covered,
        "coverage_pct": round(coverage_pct, 4),
        "similarity_threshold": sim_threshold,
        "passed": coverage_pct >= threshold,
        "invalid_lines": golden_problems + live_problems,
        "duplicate_ids": duplicate_ids,
        "uncovered_by_category": uncovered_by_cat,
        "uncovered_by_topic": uncovered_by_topic,
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
    parser.add_argument("--threshold", type=float, default=COVERAGE_THRESHOLD,
                        help="required share of covered live queries")
    parser.add_argument("--similarity-threshold", type=float, default=None,
                        help="best-match similarity that counts as covered (default: --threshold)")
    parser.add_argument("--method", choices=("embedding", "bow"), default=COVERAGE_METHOD,
                        help="similarity method (default: embedding, or COVERAGE_METHOD)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    for path in (args.golden, args.live):
        if not path.is_file():
            print(f"error: input file not found: {path}", file=sys.stderr)
            return 2

    try:
        result = run_coverage(args.golden, args.live, args.threshold, args.out,
                              similarity_threshold=args.similarity_threshold,
                              method=args.method)
    except EmbeddingUnavailable as exc:
        print(f"error: embedding similarity unavailable ({exc}). Start Ollama with the "
              "pinned model, or run with --method bow explicitly.", file=sys.stderr)
        return 2
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
