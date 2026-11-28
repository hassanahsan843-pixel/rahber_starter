# RAHBAR – Assignment 03 (RAGOps) – Final Report Draft

**Course:** Software Construction and Development (CS4032), Fall 2026  
**Assignment:** 03 – RAGOps  
**Group members:** Muhammad Ahsan Khalid (24I-3073) · Abdul-Mueed (24I-3084)  
**Repository:** https://github.com/hassanahsan843-pixel/rahber_starter  
**Branch:** `a03-dev`

---

## 1. Pipeline stage mapping (handout Figure 3)

The following maps the RAGOps pipeline stages to the implementation and Kubernetes resources in RAHBAR. The stage labels are functional descriptions; compare their wording with the exact labels in Figure 3 of the assignment handout before submission.

| Pipeline stage | RAHBAR implementation | Kubernetes responsibility |
|---|---|---|
| Ingest, extract and clean | `indexer/pipeline/extract.py`, `clean.py` | Indexer Job; `reindex` CronJob schedules ingestion |
| Verify documents | `indexer/pipeline/verify.py` | Indexer Job; verification report stored on `datalake-pvc` |
| Manage data lake, quarantine and archive | `datalake.py`, `quarantine.py`, `archive.py` | `datalake-pvc`, `quarantine-pvc`, `archive-pvc` |
| Chunk, embed and store | `chunk.py`, `embed.py`, `store.py`; `nomic-embed-text` | Chroma StatefulSet with persistent storage |
| Retrieve, cache, generate and apply guardrails | `api/app/services/answer.py`, `cache.py`, `retrieval/chroma.py` | `rahbar-api` Deployment and Redis service |
| Trace and expose metrics | `api/app/services/trace.py`, `/metrics` | API pod logs and `traces-pvc` |
| Evaluate and apply release gate | `eval/run_all.py`, `eval/release_gate.py` | `rahbar-eval` Job and `eval-thresholds` ConfigMap |
| Test and promote a candidate | `k8s/base/candidate-*.yaml` | Candidate Deployment and Service, followed by rollout/rollback |

## 2. Part A – Data management and verification

RAHBAR verifies documents before indexing them. The four implemented verification checks are:

- **Quality:** rejects empty or garbled content and extraction failures.
- **Completeness:** checks required metadata and content completeness, including truncation.
- **Recency:** compares document versions with the registry and related document versions.
- **Uniqueness:** checks exact content hashes and near-duplicate similarity using document embeddings.

Failed documents are quarantined instead of being added to the searchable collection. Superseded versions are archived so they can be restored when required.

### Batch B verification results

The cluster verification report is `docs/evidence/a03-k8s/06-batch-b-verification-report.txt`.

| Batch B document | Planted condition | Observed result |
|---|---|---|
| `academic-rules-2026-copy.md` | Exact duplicate | Quarantined as an exact duplicate |
| `fees-2026-duplicate.md` | Exact duplicate | Quarantined as an exact duplicate |
| `academic-rules-2026-paraphrased.md` | Near-duplicate | Passed but flagged as a possible near-duplicate; similarity 0.933, below the configured quarantine decision threshold of 0.97 |
| `exam-policy-truncated.txt` | Truncated content | Quarantined because it ends mid-sentence and has no usable version |
| `scholarship-policy-no-metadata.txt` | Missing metadata | Quarantined because date/version metadata is missing |
| `academic-rules-2027.md` | Newer version | Passed and identified as newer than `academic-rules-2026.md` |
| `it-policy-with-injection.md` | Prompt-injection instruction | Passed document verification; the malicious instruction was handled by the query-time guardrail |
| Hostel, library and transport documents | Valid documents | Passed verification |

The near-duplicate is an acknowledged partial result: it was detected and flagged, but not quarantined because its similarity did not reach the configured 0.97 decision threshold. The injected instruction is not a document-quality defect; it is addressed by the query-time guardrail.

### Incremental-ingestion results

- Batch B was ingested alongside Batch A. The collection increased from 24 to 70 chunks, while Batch A remained present (`03-ingestion-job.txt`, `06-batch-b-ingestion.txt`, `08-candidate-index.txt`).
- The unchanged re-run embedded **0 new chunks** (`07-unchanged-rerun.txt`).
- Changed and deleted documents are handled by document identity so that only the affected document's chunks are replaced or removed.
- Superseded versions are archived and excluded from normal retrieval through metadata filters. The archive restore test is recorded in `indexer/tests/test_batches_and_verification.py`.

## 3. Part B – Offline evaluation

The golden set in `eval/golden.jsonl` contains **44 questions**: 31 answerable questions, including 5 hard questions, and 13 questions that require abstention. The set contains 13 human-written questions and 31 synthetic questions. The synthetic answer hints were checked against source text; however, a separate record of the team's own hand-review process is not documented, so this remains a limitation.

The evaluation command is:

```powershell
python eval/run_all.py --api <API_URL>
```

The timestamped results used for this report are in `eval/results/` (including the run ending `20261128T104832Z`).

### B1 – Embedding evaluation

| Measure | Result |
|---|---:|
| Mean similarity for 15 paraphrase pairs | 0.829 |
| Mean similarity for 15 unrelated pairs | 0.432 |
| Similarity gap | 0.398 |
| Embedding time per chunk | 52.9 ms |
| Maximum determinism difference | 0.0 |

### B2 – Retrieval evaluation

| Measure | Clean questions | Typo/noise test |
|---|---:|---:|
| Recall@3 | 0.968 | 0.968 |
| Recall@5 | 0.968 | 0.968 |
| MRR | 0.887 | 0.871 |
| nDCG@5 | 0.908 | 0.896 |

### B3 – End-to-end evaluation

| Measure | Result |
|---|---:|
| Faithfulness on answered questions | 0.962 |
| Citation validity | 1.000 |
| Abstention correctness | 0.923 (12/13) |
| Answer rate | 0.839 |
| Latency p50 | 0.08 s |
| Latency p95 | 3.6 s |
| Agreement with 8 hand labels | 0.75 |

### B4 – Stochasticity

Two full runs were compared with fixed seeds and the semantic cache bypassed using `Cache-Control: no-cache`. Retrieval metric differences were zero. Faithfulness changed from 1.000 to 0.958; 47.7% of answers were identical, and five questions changed between answer and abstention (q21, q23, q24, q25 and q33). This indicates that retrieval was deterministic for the fixed index, while generated answers still varied.

### Known failure: q37

Question q37 asks whether a student may keep a mobile phone in the examination hall. The exam policy was quarantined because the available file was truncated. The retrieved top passage was instead a library rule about keeping phones silent in reading areas. The model generalised from that unrelated rule and answered rather than abstaining. A general prompt rule was tested but did not solve the issue; it was reverted because it still failed in repeated tests and reduced the answer rate from 0.903 to 0.871. The failure remains unresolved and is reported transparently.

## 4. Part C – Change experiment and release gate

The chunking experiment compared a baseline of **400 tokens / 50 overlap** with a candidate of **200 tokens / 30 overlap**, using separate collections on the real corpus. The baseline produced approximately 511–525 chunks and the candidate approximately 781–800 chunks.

The saved comparison is `eval/results/partc_gate_result.json`; the thresholds are configured in `eval/config/thresholds.yaml`.

| Metric | Baseline (400/50) | Candidate (200/30) | Candidate threshold | Candidate result |
|---|---:|---:|---:|---|
| Recall@5 | 0.923 | 0.923 | Drop no greater than 0.05 | Pass |
| Recall@3 | 0.846 | 0.846 | At least 0.50 | Pass |
| MRR | 0.647 | 0.699 | At least 0.40 | Pass |
| nDCG@5 | 0.717 | 0.756 | At least 0.40 | Pass |
| Faithfulness | 1.000 | 1.000 | At least 0.60 | Pass |
| Citation validity | 1.000 | 1.000 | At least 0.90 | Pass |
| Abstention correctness | 1.000 | 1.000 | 1.000 | Pass |
| Latency p95 | 10.9 s | 7.0 s | No more than 10 s | Pass |

The candidate's saved Part C gate result is **PASS (8/8)**. The baseline's p95 latency was 10.9 seconds, above the 10-second candidate limit; the candidate improved p95 latency to 7.0 seconds.

### In-cluster candidate and threshold demonstrations

The Kubernetes evaluation used the full golden set and produced different results from the isolated Part C chunk experiment:

| In-cluster run | Result | Explanation |
|---|---|---|
| Candidate v1 | Fail, exit code 1 | Candidate index contained Batch A only; Recall@5 fell from 0.968 to 0.290 |
| Candidate v2 | Fail, exit code 1 | Indexed Batch A and B, but abstention correctness was 0.846, below the required 1.0 because of q37 and a stochastic miss |
| ConfigMap threshold demonstration | Fail → Pass → Fail | The abstention threshold was temporarily changed to 0.8, then restored to its original value |

**Release decision:** the 200/30 chunk candidate passed the separate Part C experiment. However, neither in-cluster candidate passed the original release gate. The temporary 0.8 threshold was used only to demonstrate that the Kubernetes ConfigMap controls the gate; it is not the final release threshold and does not mean the candidate passed the original gate.

## 5. Part D – Observability and traceability

Each trace in `traces/sample_traces.jsonl` records the trace ID and timestamp, original and rewritten query, embedding model, collection, retrieved chunk IDs, document IDs and versions, similarity scores, applied filters, final chunk order, exact prompt and SHA-256 hash, model and temperature, response, citations, guardrail actions, stage latencies and cache status.

The trace tools are:

```powershell
python scripts/trace_inspect.py <TRACE_ID> --file traces/sample_traces.jsonl
python scripts/trace_summary.py --file traces/sample_traces.jsonl
```

`trace_inspect.py` reconstructs the retrieved context and prompt and checks the prompt hash. `trace_summary.py` reports top-1 similarity distribution, abstention rate, cache-hit rate, p95 stage latency and injection flags.

### Five-line root-cause analysis from trace `c046d9d8298841c1a3d495c5170f7195`

1. **Symptom:** the system answered the question about keeping a mobile phone in an examination hall, although it should have abstained.
2. **Retrieved context:** the top passage was `library-policy-2026.txt` with a score of 0.696; it discussed silent phones in reading areas, not examination halls.
3. **Generation:** the model, `llama3.2:3b`, generalised the library rule and inferred an examination-hall rule that was not present in the retrieved evidence.
4. **Root cause:** the only exam-hall policy file was quarantined as truncated, leaving the index without an authoritative answer.
5. **Corrective action:** provide and ingest a complete exam policy, then rerun q37 and the full evaluation. The attempted general prompt restriction was reverted because it did not fix the failure and reduced answer rate.

The cluster trace summary covered **1,651 traces** stored on `traces-pvc`: abstention rate 49.8%, cache-hit rate 9.2%, 117 injection flags, top-1 score minimum/mean/maximum of 0.44/0.70/0.92, and p95 latency of 45 ms for retrieval and 4.2 seconds for generation.

## 6. Part E – Live-query coverage and drift

The live set contains 40 questions: approximately 28 in-domain and 12 out-of-domain. Coverage uses cosine similarity from `nomic-embed-text`, with a similarity cutoff of 0.65 and a required coverage level of 65%.

| Golden set | Covered live queries | Coverage | Gate |
|---|---:|---:|---|
| Before expansion: 25 questions | 24/40 | 60.0% | Fail |
| After adding 19 questions | 33/40 | 82.5% | Pass |

The before-expansion result is `eval/results/coverage_before_golden_expansion.json`; the expanded result is `eval/results/coverage_report.json`. Uncovered questions are grouped by their nearest topic to guide additions to the golden set.

The expanded set improved coverage by 22.5 percentage points. A separate written analysis of Batch B drift is not present in the evidence, so the coverage improvement should not be described as a complete drift analysis.

## 7. Part F – Kubernetes deployment and demonstration

The local demonstration ran on a **kind Kubernetes v1.37 cluster** using Docker Desktop and the `k8s/overlays/local` overlay, with Ollama available on the host. It was a real Kubernetes deployment, not a Docker-only substitute. The cluster name was `rahbar-a03`.

Evidence files are stored in `docs/evidence/a03-k8s/`.

| Demonstration | Evidence | Observed result |
|---|---|---|
| Kubernetes resources and persistent storage | `05-resources.txt` | Deployments, StatefulSet, Services, PVCs, Jobs, CronJob and ConfigMaps listed |
| Initial ingestion | `03-ingestion-job.txt` | Completed after one automatic retry; 24 chunks |
| Batch B ingestion and unchanged re-run | `06-*`, `07-*` | Four documents quarantined, 46 chunks added, then zero chunks on unchanged re-run |
| Candidate collection | `08-*`, `10-*` | Separate `rahbar-candidate` collection |
| Evaluation and gate | `09-*`, `11-*` | Candidate evaluations rejected with exit code 1 |
| ConfigMap threshold edit | `12-gate-configmap-change.txt` | Gate changed FAIL → PASS → FAIL; original threshold restored |
| Rollout and rollback | `13-rollout-rollback.txt` | Rollout status and undo executed; traces show `rahbar` → `rahbar-candidate` → `rahbar` |
| Metrics and traces | `14-*` | Metrics reachable through port-forward; 1,651 traces on `traces-pvc` |

The rollout was a demonstration of the Kubernetes promotion and rollback mechanism under the temporary demo threshold. It must not be interpreted as an approval under the original release gate, because candidate v2 failed the original abstention threshold.

The demonstration cluster was kind rather than Minikube. If the instructor specifically requires Minikube, this difference must be disclosed.

## 8. RAGOps qualities from handout Table 2

The implementation addresses the five qualities as follows. Use the exact wording from Table 2 in the handout if it differs from the labels below.

| RAGOps quality | How it is addressed in RAHBAR | Evidence / measurement |
|---|---|---|
| Adaptability | Chunk size and overlap can be changed and compared using separate collections and the evaluation suite. | Part C: 400/50 compared with 200/30; candidate passed 8/8 in the separate experiment. |
| Monitorability | Operational metrics and evaluation outputs expose system behaviour and performance. | `/metrics`, evaluation result files, p50/p95 latency and trace summary. |
| Observability | Traces record retrieval, generation, guardrails, cache status and per-stage latency. | 1,651 cluster traces; `trace_inspect.py` and `trace_summary.py`. |
| Traceability | Each answer can be linked to its query, retrieved documents and versions, prompt and prompt hash, response and citations. | `traces/sample_traces.jsonl`; prompt reconstruction and SHA-256 integrity check. |
| Reliability | Verification, quarantine, idempotent ingestion, release gates, persistent storage and rollback reduce operational risk. | Batch B verification; zero-chunk unchanged run; release-gate results; PVCs; rollout undo. |

## 9. Short questions

### (a) Which Batch B issues did verification catch, and what did it miss?

Verification caught the two exact duplicates, the truncated exam policy and the document with missing date/version metadata. It also recognised the 2027 academic rules as a newer version. The paraphrased near-duplicate was detected and flagged at similarity 0.933, but it was not quarantined because it did not reach the configured 0.97 decision threshold. The injected instruction passed document verification because it is not a file-quality defect; it was handled by the query-time guardrail. A more complete near-duplicate policy could use a reviewed threshold and a manual-review band, rather than automatically lowering the threshold without testing false positives.

### (b) Did a module-level metric look good while a component or end-to-end result did not?

Yes. B1 showed a useful embedding separation: mean similarity was 0.829 for paraphrases and 0.432 for unrelated pairs, with a gap of 0.398. Retrieval metrics were also high, with clean Recall@5 of 0.968. However, the end-to-end abstention result was only 12/13 (0.923), because q37 was answered using an unrelated library rule after the correct exam policy had been quarantined. This demonstrates that good embeddings and retrieval scores do not guarantee that the final answer is supported or that the system abstains correctly.

### (c) What breaks if old document versions are deleted instead of archived?

Deleting old versions removes the ability to restore a previous version after a bad update, investigate historical answers, or reproduce which document version supported an earlier response. It also weakens auditability because traces may refer to a document version that no longer exists. Archiving preserves rollback and historical provenance while allowing the current version to remain the normal retrieval target.

### (d) Which metric is least trusted, and why?

The token-overlap faithfulness measure is less reliable than a careful human evidence review. A response can share words with a passage while still making an unsupported inference, and a correct paraphrase may use different words. The limited judge comparison also had 0.75 agreement with eight hand labels, so the faithfulness score should be treated as an estimate rather than proof that every answer is grounded.

### (e) Which RAGOps idea was hardest to implement, and what changed during implementation?

The most difficult part was making evaluation and release decisions operational in Kubernetes rather than leaving them as local scripts. The implementation needed separate candidate indexing, evaluation Jobs, configurable thresholds, and a rollout/rollback demonstration. The cluster tests also exposed a candidate that was missing Batch B and later the unresolved q37 abstention failure. The final result therefore distinguishes the passing isolated chunk experiment from the candidates that failed the original in-cluster gate.

### (f) Which Kubernetes object carries each responsibility? Which object controls writes to the vector store?

The API Deployment serves user requests, while the Chroma StatefulSet runs the vector database with persistent storage. Services provide stable network access. PVCs persist the data lake, quarantine, archive, Chroma data and traces. The ingestion Job performs verification and indexing; the reindex CronJob schedules that Job. The evaluation Job runs the test suite and release gate. The ConfigMap supplies non-secret evaluation thresholds, while Kubernetes Secrets hold credentials. A separate candidate Deployment and Service isolate candidate testing. The intended vector-store write path is the ingestion Job (including Jobs created by the reindex CronJob); the API is a reader/query service, not the ingestion writer.

## 10. Contributions and AI-use statement

| Member | Contribution |
|---|---|
| **Muhammad Ahsan Khalid (24I-3073)** | Integration and final merge on `a03-dev`; evaluation and release-gate work; coverage work; Kubernetes integration and demonstration evidence; final report and submission preparation. |
| **Abdul-Mueed (24I-3084)** | Data verification and ingestion work (Part A); tracing/observability work (Part D); related implementation and testing. |

The repository contribution summary previously recorded approximately **62.3% for Ahsan and 37.7% for Abdul-Mueed**. Confirm these percentages against the final GitHub contribution view before submission; they are not a substitute for the instructor's required contribution explanation.

**AI-use statement:** AI assistants were used to support code review, debugging, test development and report drafting. The team reviewed the suggested changes, ran the implementation and tests, and used the repository's actual result files and Kubernetes evidence for the reported measurements. The team remains responsible for the submitted code, results and explanations.

## 11. Appendix – screenshots

Insert the team's own uncropped terminal screenshots here. Each screenshot should show the command and its output, with a short caption. Keep screenshots in the appendix so the main report remains within the eight-page limit.

1. **A – Batch B verification:** verification report showing pass/fail reasons and quarantined documents.
2. **A – Incremental ingestion:** unchanged ingestion showing zero new chunks embedded.
3. **B – Evaluation:** B1 embedding, B2 retrieval and B3 end-to-end evaluation summary.
4. **C – Release gate:** one failing candidate and the passing separate Part C chunk experiment; label the two contexts clearly.
5. **D – Trace reconstruction:** full trace and reconstructed context/prompt, plus trace summary.
6. **E – Coverage:** before and after coverage results (60.0% and 82.5%).
7. **F – Kubernetes resources:** `kubectl get all,pvc,configmap,cronjob,job -n rahbar`.
8. **F – Kubernetes Jobs:** completed ingestion and evaluation Job logs.
9. **F – ConfigMap gate demonstration:** FAIL → PASS → FAIL, with the temporary 0.8 threshold clearly labelled and original value restored.
10. **F – Rollout and rollback:** `kubectl rollout status`, `kubectl rollout undo`, and trace showing the collection returned to `rahbar`.

---

**Final submission checks:** verify the exact Figure 3 and Table 2 wording against the handout; confirm both member details and contribution percentages; insert genuine screenshots; export to PDF with no more than eight main pages; and ensure the PDF and evidence are committed to `a03-dev`.
