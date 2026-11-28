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
| `academic-rules-2026-paraphrased.md` | **Near-duplicate** of academic-rules-2026.md (paraphrased, same meaning) | Identified as a near-duplicate in the report (embedding similarity 0.933); quarantined only at or above `NEAR_DUPLICATE_THRESHOLD` (0.97) – see note 2 |

## Notes

1. `academic-rules-2026-copy.md` and `fees-2026-duplicate.md` will produce
   identical content hashes to their counterparts in `indexer/corpus/`.
   The uniqueness checker catches this via doc_hash comparison against the
   data lake registry.

2. `academic-rules-2026-paraphrased.md` is semantically identical to the
   2026 rules but uses different wording. The near-duplicate check compares
   whole-document embeddings from the pinned model (`nomic-embed-text`):
   - paraphrase vs 2026 rules: **0.933** → reported in the verification report as
     "WARNING possible near-duplicate" (report band starts at
     `NEAR_DUPLICATE_REPORT_THRESHOLD`, 0.90);
   - it is **not quarantined**, because the configured quarantine threshold
     `NEAR_DUPLICATE_THRESHOLD` is 0.97. Setting it to 0.93 or lower would
     quarantine it; the threshold was not changed to force that result;
   - versions of the same document (`academic-rules-2023/-2026/-2027`, similarity
     up to 0.97) are skipped by this check: the recency checks handle versions;
   - the highest similarity between distinct, non-duplicate documents is 0.84.
   With bag-of-words vectors (offline CI) the paraphrase scores 0.81 and is not flagged.

3. `it-policy-with-injection.md` passes verification (quality, completeness,
   recency, uniqueness all pass) but the injected instruction `IGNORE ALL
   PREVIOUS INSTRUCTIONS` appears in the retrieved text. The guardrail in
   the API must NOT obey this instruction. The trace records the guardrail
   action.

4. `exam-policy-truncated.txt` ends mid-sentence. The truncation check flags a
   `.md`/`.txt` document whose last line is prose that stops without closing
   punctuation; list items, table rows and `Key: value` lines are not flagged.

5. `scholarship-policy-no-metadata.txt` has no YAML front matter, so title
   and date/version cannot be extracted. (`library-policy-2026.txt` has no front
   matter either, but its file name carries the year, which is used as its version.)

6. Data correction (2026-11-28): `academic-rules-2026-copy.md` and
   `fees-2026-duplicate.md` were regenerated as byte-identical copies of the
   Batch A files, and `academic-rules-2026-paraphrased.md` as a paraphrase of the
   current Batch A rules. The earlier versions were drafted against a different
   source document (different institution, different fees), so they were not
   duplicates of Batch A and contradicted it. The previous files remain in Git history.
