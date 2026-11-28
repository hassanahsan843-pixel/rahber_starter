#!/usr/bin/env python3
"""B5 – One-command evaluation runner.

Runs the complete evaluation suite:
  1. B1: Embedding evaluation
  2. B2: Retrieval evaluation (clean + noise)
  3. B3: End-to-end evaluation
  4. B4: Stochasticity comparison
  5. Summary table
  6. Saves timestamped result file under eval/results/

Usage:
    python eval/run_all.py --api http://127.0.0.1:8000
    python eval/run_all.py --api http://127.0.0.1:8000 --skip-embed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def print_banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_all(api: str, golden_path: Path, skip_embed: bool = False,
            seed: int = 42, out_path: Path | None = None) -> dict:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {"timestamp": stamp, "api": api}

    # B1: Embedding
    if not skip_embed:
        print_banner("B1 – Embedding Evaluation")
        from eval.embed_eval import run_embed_eval
        embed_out = results_dir / f"embed_{stamp}.json"
        all_results["embed"] = run_embed_eval(out_path=embed_out)
    else:
        print("\n[Skipping B1 embedding eval (--skip-embed)]")
        all_results["embed"] = None

    # B2: Retrieval
    print_banner("B2 – Retrieval Evaluation (clean + noise)")
    from eval.retrieval_eval import run_retrieval_eval
    ret_out = results_dir / f"retrieval_{stamp}.json"
    all_results["retrieval"] = run_retrieval_eval(api, golden_path, out_path=ret_out, seed=seed)

    # B3: End-to-end
    print_banner("B3 – End-to-End Evaluation")
    from eval.e2e_eval import run_e2e_eval
    e2e_out = results_dir / f"e2e_{stamp}.json"
    all_results["e2e"] = run_e2e_eval(api, golden_path, out_path=e2e_out)

    # B4: Stochasticity
    print_banner("B4 – Stochasticity Check")
    from eval.stochasticity_eval import run_stochasticity_eval
    stoch_out = results_dir / f"stochasticity_{stamp}.json"
    all_results["stochasticity"] = run_stochasticity_eval(api, golden_path, out_path=stoch_out)

    # Summary table
    print_banner("SUMMARY TABLE")
    ret_base = all_results["retrieval"]["baseline"]
    e2e = all_results["e2e"]
    embed = all_results.get("embed") or {}

    rows = [
        ("B1 Paraphrase similarity",  str(embed.get("mean_paraphrase_similarity", "N/A"))),
        ("B1 Unrelated similarity",   str(embed.get("mean_unrelated_similarity", "N/A"))),
        ("B1 Similarity gap",         str(embed.get("similarity_gap", "N/A"))),
        ("B1 Time per chunk (ms)",    str(embed.get("time_per_chunk_ms", "N/A"))),
        ("B2 Recall@3 (clean)",       str(ret_base.get("recall_at_3"))),
        ("B2 Recall@5 (clean)",       str(ret_base.get("recall_at_5"))),
        ("B2 MRR (clean)",            str(ret_base.get("mrr"))),
        ("B2 nDCG@5 (clean)",         str(ret_base.get("ndcg_at_5"))),
        ("B3 Faithfulness",           str(e2e.get("faithfulness_mean"))),
        ("B3 Citation validity",      str(e2e.get("citation_validity_mean"))),
        ("B3 Abstention correctness", str(e2e.get("abstention_correctness"))),
        ("B3 Latency p50 (s)",        str(e2e.get("latency_p50_s"))),
        ("B3 Latency p95 (s)",        str(e2e.get("latency_p95_s"))),
    ]

    col_w = 32
    print(f"\n  {'Metric':<{col_w}}  {'Value'}")
    print(f"  {'-'*col_w}  {'-'*10}")
    for name, val in rows:
        print(f"  {name:<{col_w}}  {val}")

    # Save combined result file
    combined_out = out_path or results_dir / f"all_{stamp}.json"
    combined_out.parent.mkdir(parents=True, exist_ok=True)
    combined_out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n  Combined results saved to {combined_out}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B5: Full evaluation suite")
    parser.add_argument("--api", required=True)
    # EVAL_GOLDEN lets an overlay pick the golden set without editing the Job command:
    # golden.jsonl targets the synthetic corpus, golden.yaml the real (prod PVC) corpus.
    parser.add_argument("--golden", type=Path,
                        default=Path(os.environ.get("EVAL_GOLDEN") or HERE / "golden.jsonl"))
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip B1 embedding eval (no indexer virtualenv needed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_all(args.api, args.golden, skip_embed=args.skip_embed,
        seed=args.seed, out_path=args.out)
