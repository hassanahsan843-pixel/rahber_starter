# Batch B – Test Corpus Documentation

**Purpose:** Batch B is used to test Part A verification, Part B drift evaluation,
and Part E coverage experiments. Each document has a planted problem clearly
described here.

## Document Inventory

| File | Planted Problem | Expected Verification Outcome |
|------|-----------------|-------------------------------|
| `academic-rules-2027.md` | **Newer version** of academic-rules-2026.md | PASS – newer version replaces old |
| `academic-rules-2026-copy.md` | **Exact duplicate** of `indexer/corpus/academic-rules-2026.md` | QUARANTINE – exact duplicate (same hash) |
| `fees-2026-duplicate.md` | **Exact duplicate** of `indexer/corpus/fees-2026.md` | QUARANTINE – exact duplicate |
| `hostel-rules-2026.md` | Valid new document | PASS |
| `library-policy-2026.txt` | Valid new document | PASS |
| `scholarship-policy-no-metadata.txt` | **Missing metadata** – no front matter, no title, no date | QUARANTINE – completeness failure |
| `exam-policy-truncated.txt` | **Truncated document** – ends mid-sentence | QUARANTINE – completeness_truncation failure |
| `it-policy-with-injection.md` | **Injected instruction** in document body | PASS (content passes; injection detected by guardrail at query time) |
| `transport-routes-2026.md` | Valid new document | PASS |
| `academic-rules-2026-paraphrased.md` | **Near-duplicate** of academic-rules-2026.md (paraphrased, same meaning) | QUARANTINE – near-duplicate detected |

## Notes

1. `academic-rules-2026-copy.md` and `fees-2026-duplicate.md` will produce
   identical content hashes to their counterparts in `indexer/corpus/`.
   The uniqueness checker catches this via doc_hash comparison against the
   data lake registry.

2. `academic-rules-2026-paraphrased.md` is semantically identical to the
   2026 rules but uses different wording. The near-duplicate checker uses
   bag-of-words cosine similarity; with threshold 0.97 this may or may not
   be caught depending on vocabulary overlap. This is intentional – it tests
   the threshold.

3. `it-policy-with-injection.md` passes verification (quality, completeness,
   recency, uniqueness all pass) but the injected instruction `IGNORE ALL
   PREVIOUS INSTRUCTIONS` appears in the retrieved text. The guardrail in
   the API must NOT obey this instruction. The trace records the guardrail
   action.

4. `exam-policy-truncated.txt` ends mid-sentence. The truncation heuristic
   checks whether the last page ends abruptly (< 20 chars, no terminal punctuation).

5. `scholarship-policy-no-metadata.txt` has no YAML front matter, so title
   and date/version cannot be extracted.
