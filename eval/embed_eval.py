#!/usr/bin/env python3
"""B1 – Module-level embedding evaluation.

Measures:
  1. Mean cosine similarity for paraphrase pairs (should be high)
  2. Mean cosine similarity for unrelated pairs (should be low)
  3. Similarity gap between groups
  4. Embedding time per chunk
  5. Determinism: embed same text twice, report max difference

Uses the same embedding function as the indexer pipeline.
Run from the repo root:
    python eval/embed_eval.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "indexer"))


def _get_embed_fn():
    """Return the embed function from the indexer pipeline."""
    from pipeline.config import get_settings
    from pipeline.embed import embed as _embed

    settings = get_settings()

    def embed_texts(texts: list[str]) -> list[list[float]]:
        return _embed(texts, settings)

    return embed_texts, os.environ.get("EMBED_MODEL", settings.embed_model)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (n1 * n2)))


# ---------------------------------------------------------------------------
# Paraphrase pairs (at least 15) – same meaning, different wording
PARAPHRASE_PAIRS = [
    ("What is the minimum attendance required?",
     "How much attendance must a student maintain?"),
    ("Students barred from exam if attendance falls below 80%.",
     "Students with less than 80% attendance cannot sit the final examination."),
    ("A CGPA below 2.00 triggers Academic Warning.",
     "Academic Warning is issued when CGPA drops under 2.00."),
    ("When repeating a course only the most recent grade counts.",
     "The latest grade is used when a student retakes a course."),
    ("The BS tuition fee is Rs. 10,000 per credit hour.",
     "Each credit hour costs Rs. 10,000 for BS students."),
    ("Hostel curfew on weekdays is 10 PM.",
     "Residents must return to the hostel by 10:00 PM on weekdays."),
    ("The library loan period is 14 days.",
     "Books borrowed from the library must be returned within two weeks."),
    ("Merit scholarships require a CGPA of 3.5 or above.",
     "Students with CGPA at least 3.5 are eligible for merit-based scholarships."),
    ("University transport routes connect Rawalpindi and Islamabad to campus.",
     "Bus services run between FAST-NUCES and both Rawalpindi and Islamabad."),
    ("Mobile phones are banned from examination halls.",
     "Electronic devices including phones must not be brought to exams."),
    ("All devices must be registered with the IT department.",
     "Students must register their personal devices before connecting to the network."),
    ("A student dropped from the program for sustained low GPA.",
     "Continued Academic Warning across two semesters leads to program dismissal."),
    ("Library fines are Rs. 10 per overdue book per day.",
     "Each overdue book incurs a daily fine of ten rupees."),
    ("University email must be used for official communication.",
     "Official correspondence with faculty requires the institutional email account."),
    ("Minimum 130 credit hours needed for the BS degree.",
     "To graduate with a BS, students must complete at least 130 credit hours."),
]

# Unrelated pairs (at least 15) – different topics
UNRELATED_PAIRS = [
    ("What is the minimum attendance required?",
     "The library loan period is 14 days."),
    ("A CGPA below 2.00 triggers Academic Warning.",
     "Mobile phones are banned from examination halls."),
    ("The BS tuition fee is Rs. 10,000 per credit hour.",
     "Hostel curfew on weekdays is 10 PM."),
    ("Merit scholarships require a CGPA of 3.5 or above.",
     "All devices must be registered with the IT department."),
    ("University transport routes connect Rawalpindi to campus.",
     "Library fines are Rs. 10 per overdue book per day."),
    ("When repeating a course only the most recent grade counts.",
     "University transport routes connect Rawalpindi to campus."),
    ("Hostel curfew on weekdays is 10 PM.",
     "Merit scholarships require a CGPA of 3.5 or above."),
    ("Mobile phones are banned from examination halls.",
     "The BS tuition fee is Rs. 10,000 per credit hour."),
    ("Library loan period is 14 days with one renewal.",
     "A CGPA below 2.00 triggers Academic Warning."),
    ("University email must be used for official communication.",
     "Hostel guests require prior written approval from the warden."),
    ("Minimum 130 credit hours needed for the BS degree.",
     "Library fines are Rs. 10 per overdue book per day."),
    ("Hostel guests require prior written approval from the warden.",
     "The BS tuition fee is Rs. 10,000 per credit hour."),
    ("Bus departure time is 7:30 AM from designated stops.",
     "A CGPA below 2.00 triggers Academic Warning."),
    ("Students may appeal results within 10 working days.",
     "Library loan period is 14 days with one renewal."),
    ("Data security: do not share university credentials.",
     "Hostel curfew on weekdays is 10 PM."),
]


def run_embed_eval(out_path: Path | None = None) -> dict:
    embed_fn, model_name = _get_embed_fn()

    print(f"\n=== B1: Embedding Evaluation (model: {model_name}) ===\n")

    # Collect all unique texts
    all_texts: list[str] = []
    for a, b in PARAPHRASE_PAIRS + UNRELATED_PAIRS:
        if a not in all_texts:
            all_texts.append(a)
        if b not in all_texts:
            all_texts.append(b)

    # Measure embed time
    t0 = time.perf_counter()
    all_vectors = embed_fn(all_texts)
    total_time = time.perf_counter() - t0
    time_per_chunk = total_time / len(all_texts) if all_texts else 0.0

    vec_map = {t: v for t, v in zip(all_texts, all_vectors)}

    # Paraphrase similarities
    para_sims: list[float] = []
    for a, b in PARAPHRASE_PAIRS:
        sim = cosine_similarity(vec_map[a], vec_map[b])
        para_sims.append(sim)
        print(f"  [paraphrase] sim={sim:.4f}  '{a[:50]}'")

    mean_para = sum(para_sims) / len(para_sims)

    # Unrelated similarities
    unrel_sims: list[float] = []
    for a, b in UNRELATED_PAIRS:
        sim = cosine_similarity(vec_map[a], vec_map[b])
        unrel_sims.append(sim)

    mean_unrel = sum(unrel_sims) / len(unrel_sims)
    gap = mean_para - mean_unrel

    # Determinism: embed same text twice
    test_text = PARAPHRASE_PAIRS[0][0]
    v1 = embed_fn([test_text])[0]
    v2 = embed_fn([test_text])[0]
    max_diff = max(abs(a - b) for a, b in zip(v1, v2))

    print(f"\n  Mean paraphrase similarity : {mean_para:.4f}")
    print(f"  Mean unrelated similarity  : {mean_unrel:.4f}")
    print(f"  Similarity gap             : {gap:.4f}")
    print(f"  Time per chunk             : {time_per_chunk*1000:.1f} ms")
    print(f"  Determinism max difference : {max_diff:.8f}")

    result = {
        "model": model_name,
        "mean_paraphrase_similarity": round(mean_para, 4),
        "mean_unrelated_similarity": round(mean_unrel, 4),
        "similarity_gap": round(gap, 4),
        "time_per_chunk_ms": round(time_per_chunk * 1000, 2),
        "determinism_max_diff": max_diff,
        "paraphrase_pairs": len(PARAPHRASE_PAIRS),
        "unrelated_pairs": len(UNRELATED_PAIRS),
    }

    if out_path:
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  Results saved to {out_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="B1: Embedding evaluation")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_embed_eval(out_path=args.out)
