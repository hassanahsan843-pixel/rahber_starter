# RAHBAR Indexing and Retrieval Pipeline Report

> Measurement note: these chunking and cache-threshold figures are carried forward from the prior project work. They were not reproduced during the current hardening pass; rerun them against the live cluster and current corpus before treating them as verified results.

This document reports the empirical measurements, configuration tuning, extraction challenges, and semantic caching thresholds across the 5-stage indexing and query pipeline.

---

## 1. Stage 1: Extraction & Cleaning Analysis

### Documents Requiring OCR
- **Document**: `Faculty Entrepreneurship and Technical Start-ups Policy` (`27.pdf?20220530`, 4 pages).
- **Diagnosis**: Running `pdffonts` returned zero font entries, confirming that the document was a flat raster scan with no embedded text layer.
- **Handling**: `extract.py` detects pages where extracted characters are under 30 and routes the page image through `pytesseract` OCR.
- **Recovery Limitation**: Hand-written signatures and rubber stamps in the margins could not be resolved cleanly into ASCII text, but all critical operational clauses (e.g., 30% university equity participation) were successfully recovered.

### Table Extraction: Before and After
In the university `Fee Structure` document and HTML tables, naive text extraction produced scrambled lines:

#### Before (Mangled Extraction)
```text
Tuition Fee Admission Charges Rs. 11,000 per credit hour Rs. 35,000 Bachelor of Science Master of Science
100% Refund 50% Refund 1st week 2nd week
```
Column relationships and numeric amounts were completely decoupled from degree programs.

#### After (Structured Markdown Table in Extraction)
```markdown
| Program / Fee Type | Rate | Description |
| --- | --- | --- |
| BS Tuition Fee | Rs. 11,000 | Per credit hour |
| Admission Fee | Rs. 35,000 | One-time non-refundable |
| Tuition Refund (1st week) | 100% | Full refund minus admission charges |
| Tuition Refund (2nd week) | 50% | Half refund |
| Tuition Refund (3rd week+) | 0% | No refund |
```
Structure-aware HTML parsing in `_SimpleHTMLTableParser` and markdown pipe formatting preserved table headers, allowing the retriever to return structured tabular context.

---

## 2. Stage 3: Structure-Aware Chunking Measurements

We evaluated three different (chunk_size, chunk_overlap) configurations over the 15-question golden evaluation set (`eval/golden.yaml`), measuring retrieval Recall@5 and storage amplification.

### Chunking Configurations Table
| Configuration | Window Size (words) | Overlap (words) | Recall@5 | Storage Amplification (Stored / Original Tokens) | Qualitative Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Config A (Small Window)** | 150 | 25 | 0.7850 | 1.18x (118k / 100k) | Cut long definitions in half; split Generative AI policy across 4 chunks; fragmented clause numbers. |
| **Config B (Shipped Configuration)** | **400** | **50** | **0.9231** | **1.14x (114k / 100k)** | **Optimal balance. Kept entire policy sections intact; preserved table structures; optimal Recall@5.** |
| **Config C (Large Window)** | 800 | 100 | 0.8120 | 1.12x (112k / 100k) | Introduced vector dilution; noise from surrounding paragraphs degraded cosine similarity for exact-term queries. |

### Justification of Shipped Setting
Configuration B (`size = 400`, `overlap = 50`) was shipped. It achieved the highest Recall@5 (0.9231 / 92.3%) and highest citation validity (0.9850 / 98.5%). The 50-word overlap provides a 1.14x storage amplification ratio (comparable to the lecture's 150/120 benchmark), ensuring boundary conditions between paragraphs are never lost while avoiding vector representation dilution.

---

## 3. Stage 4: Semantic Cache Threshold Selection

The semantic cache checks query similarity against cached turn vectors before invoking retrieval or LLM generation. Section 3.3 establishes the governing rule: **select the lowest similarity threshold with zero false hits on test questions**.

We tested three candidate cosine similarity thresholds:
1. **Threshold 0.75**: 
   - *Result*: Rejected due to false hits.
   - *False Hit Example*: The question *"What is the CGPA required to avoid academic warning?"* (Answer: 2.00) incorrectly matched the cached question *"What is the SGPA required to avoid probation in the first semester?"* (Answer: 1.50) at similarity 0.78. Serving this cached answer gave incorrect academic standing criteria.
2. **Threshold 0.82**: 
   - *Result*: Rejected due to false hits.
   - *False Hit Example*: The question *"Can a student conditionally register after failing a prerequisite course?"* incorrectly matched *"What is the procedure to repeat an elective course?"* at similarity 0.84.
3. **Threshold 0.88 (Shipped)**:
   - *Result*: **Shipped with zero false hits**. Paraphrased variations of identical questions (e.g., *"How much attendance is required to sit in final exams?"* vs *"What is the mandatory attendance percentage for finals?"*) match cleanly at similarity 0.91, while distinct academic clauses remain safely segregated.

### Never-Cache Guardrails & Invalidation
- **Never Cache**: Follow-up questions, abstentions (`answered_from_context: false`), and low-confidence responses are strictly excluded from the cache.
- **Cache Invalidation**: Whenever the indexing Job runs, `cache.invalidate()` increments `generation_id` and flushes Redis keys via `FLUSHDB`, ensuring stale answers are never served across re-indexing cycles.
