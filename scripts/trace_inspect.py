#!/usr/bin/env python3
"""D2 – Trace inspection CLI.

Accepts a trace_id (request_id), reads the trace from traces.jsonl,
and reconstructs the full pipeline view:
  - Original query
  - Retrieved context
  - Applied filters
  - Prompt hash (and reconstructed prompt)
  - Model and parameters
  - Final response
  - Citations
  - Guardrail actions

Usage:
    python scripts/trace_inspect.py <trace_id>
    python scripts/trace_inspect.py <trace_id> --file /traces/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_TRACE_FILE = os.environ.get("TRACE_FILE", "/traces/traces.jsonl")


def find_trace(trace_id: str, trace_file: str) -> dict | None:
    path = Path(trace_file)
    if not path.exists():
        print(f"ERROR: Trace file not found: {trace_file}", file=sys.stderr)
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if record.get("trace_id") == trace_id:
                return record
        except Exception:
            continue
    return None


def print_trace(record: dict) -> None:
    sep = "─" * 60

    print(f"\n{'=' * 60}")
    print(f"  TRACE: {record.get('trace_id')}")
    print(f"  Time:  {record.get('timestamp')}")
    print(f"{'=' * 60}\n")

    print(f"  QUERY\n  {sep}")
    print(f"  {record.get('query')}\n")

    if record.get("query_rewritten"):
        print(f"  REWRITTEN QUERY\n  {sep}")
        print(f"  {record.get('query_rewritten')}\n")

    print(f"  EMBEDDING MODEL:  {record.get('embed_model')}")
    print(f"  COLLECTION:       {record.get('collection')}\n")

    filters = record.get("filters_applied") or {}
    print(f"  FILTERS APPLIED\n  {sep}")
    if filters:
        for k, v in filters.items():
            print(f"  {k}: {v}")
    else:
        print("  (none)")
    print()

    chunks = record.get("retrieved_chunks") or []
    print(f"  RETRIEVED CONTEXT ({len(chunks)} chunks)\n  {sep}")
    for i, c in enumerate(chunks, 1):
        doc = f" doc={c.get('doc_id')} v{c.get('doc_version')}" if c.get("doc_id") else ""
        sup = " SUPERSEDED" if c.get("superseded") else ""
        print(f"  [{i}] {c.get('chunk_id')} {c.get('source')} / {c.get('section')} "
              f"(p.{c.get('page')})  score={c.get('score')}{doc}{sup}")
    if record.get("reranked_order"):
        print(f"  Final order: {', '.join(record['reranked_order'])}")
    print()

    print(f"  LLM MODEL:        {record.get('llm_model')}")
    print(f"  TEMPERATURE:      {record.get('llm_temperature')}")
    print(f"  PROMPT HASH:      {record.get('prompt_hash')}")
    prompt = record.get("prompt")
    if prompt is not None:
        import hashlib

        ok = hashlib.sha256(prompt.encode("utf-8")).hexdigest() == record.get("prompt_hash")
        print(f"  PROMPT CHECK:     {'reconstructed prompt matches hash' if ok else 'HASH MISMATCH'}")
    print(f"  PROVIDER:         {record.get('provider')}")
    print(f"  CACHE HIT:        {record.get('cache_hit')}\n")

    if prompt:
        print(f"  EXACT PROMPT SENT TO THE MODEL (system + user)\n  {sep}")
        for line in prompt.splitlines():
            print(f"  | {line}")
        print()
    elif prompt == "":
        print("  (no model call: answered from cache or abstained before generation)\n")

    print(f"  RESPONSE\n  {sep}")
    print(f"  {record.get('response')}\n")

    cits = record.get("citations") or []
    print(f"  CITATIONS ({len(cits)})\n  {sep}")
    for c in cits:
        page = f"p.{c['page']}" if c.get("page") else ""
        print(f"  • {c.get('source')} / {c.get('section')} {page}")
    print()

    print(f"  ANSWERED FROM CONTEXT:  {record.get('answered_from_context')}")
    print(f"  CONFIDENCE:             {record.get('confidence')}\n")

    guards = record.get("guardrail_actions") or []
    print(f"  GUARDRAIL ACTIONS ({len(guards)})\n  {sep}")
    for g in guards:
        print(f"  ⚠  {g}")
    if not guards:
        print("  (none)")
    print()

    lat = record.get("latency") or {}
    print(f"  LATENCY\n  {sep}")
    print(f"  Retrieval:   {lat.get('retrieval_ms')} ms")
    print(f"  Generation:  {lat.get('generation_ms')} ms")
    print(f"  Cache:       {lat.get('cache_lookup_ms')} ms")
    print(f"  Total:       {lat.get('total_ms')} ms\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="D2: Trace inspection tool")
    parser.add_argument("trace_id", help="trace_id (request_id) to inspect")
    parser.add_argument("--file", default=DEFAULT_TRACE_FILE,
                        help=f"Path to traces.jsonl (default: {DEFAULT_TRACE_FILE})")
    args = parser.parse_args()

    record = find_trace(args.trace_id, args.file)
    if record is None:
        print(f"Trace '{args.trace_id}' not found in {args.file}", file=sys.stderr)
        return 1

    print_trace(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
