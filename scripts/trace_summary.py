#!/usr/bin/env python3
"""D3 – Trace summary script.

Summarizes traces.jsonl over a selected time window.
Reports:
  - Top-1 retrieval similarity distribution
  - Abstention rate
  - Cache hit rate
  - p95 latency per pipeline stage
  - Number of flagged prompt-injection attempts

Usage:
    python scripts/trace_summary.py
    python scripts/trace_summary.py --hours 24 --file /traces/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

DEFAULT_TRACE_FILE = os.environ.get("TRACE_FILE", "/traces/traces.jsonl")


def load_traces(trace_file: str, hours: float = 24.0) -> list[dict]:
    path = Path(trace_file)
    if not path.exists():
        return []
    cutoff = time.time() - hours * 3600
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            # Parse ISO timestamp
            ts_str = rec.get("timestamp", "")
            if ts_str:
                import datetime
                ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.timestamp() >= cutoff:
                    records.append(rec)
        except Exception:
            continue
    return records


def summarize(records: list[dict]) -> None:
    if not records:
        print("  No traces found in the selected time window.")
        return

    n = len(records)
    abstentions = sum(1 for r in records if not r.get("answered_from_context", True))
    cache_hits = sum(1 for r in records if r.get("cache_hit", False))
    injections = sum(1 for r in records if r.get("guardrail_actions"))

    # Top-1 retrieval score distribution
    top1_scores = []
    for r in records:
        chunks = r.get("retrieved_chunks") or []
        if chunks:
            scores = [c.get("score", 0) for c in chunks]
            top1_scores.append(max(scores) if scores else 0)

    # Latency percentiles
    retrieval_lats = [r["latency"]["retrieval_ms"] for r in records if r.get("latency")]
    gen_lats = [r["latency"]["generation_ms"] for r in records if r.get("latency")]
    total_lats = [r["latency"]["total_ms"] for r in records if r.get("latency")]

    def p95(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[int(len(s) * 0.95)]

    print(f"\n=== D3: Trace Summary ({n} queries) ===\n")
    print(f"  Abstention rate          : {abstentions/n:.1%} ({abstentions}/{n})")
    print(f"  Cache hit rate           : {cache_hits/n:.1%} ({cache_hits}/{n})")
    print(f"  Injection flags          : {injections}")
    print()
    if top1_scores:
        print(f"  Top-1 retrieval score")
        print(f"    min  : {min(top1_scores):.4f}")
        print(f"    mean : {statistics.mean(top1_scores):.4f}")
        print(f"    max  : {max(top1_scores):.4f}")
    print()
    print(f"  Latency p95 (ms)")
    print(f"    retrieval  : {p95(retrieval_lats):.1f}")
    print(f"    generation : {p95(gen_lats):.1f}")
    print(f"    total      : {p95(total_lats):.1f}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="D3: Trace summary")
    parser.add_argument("--file", default=DEFAULT_TRACE_FILE)
    parser.add_argument("--hours", type=float, default=24.0,
                        help="Summarize traces from the last N hours (default: 24)")
    args = parser.parse_args()

    records = load_traces(args.file, args.hours)
    summarize(records)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
