# A03 Compliance Checklist – RAHBAR RAGOps

**Assignment:** SCD Assignment 03 – RAGOps  
**Branch:** `a03-dev`  
**Team:** hassanahsan843-pixel, Abdul-Mueed92

---

## Part A – Data Management

### A1. Document Verification

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Quality check: empty document detection | ✅ | `indexer/pipeline/verify.py` → `_check_quality()` |
| Quality check: garbled/unusable text detection | ✅ | `verify.py` entropy check + non-alpha ratio |
| Quality check: extraction failure detection | ✅ | `verify.py` checks empty pages list |
| Completeness: source metadata required | ✅ | `verify.py` → `_check_completeness()` |
| Completeness: title required | ✅ | `verify.py` checks `doc.title` |
| Completeness: date/version required | ✅ | `verify.py` checks `doc.date` and `doc.doc_version` |
| Completeness: page count required | ✅ | `verify.py` checks `doc.page_count >= 1` |
| Completeness: truncation detection | ✅ | `verify.py` checks text ends abruptly mid-sentence |
| Recency: detect newer version of existing doc | ✅ | `verify.py` → `_check_recency()` via data lake lookup |
| Recency: identify which version is newer | ✅ | Semver/date comparison in `_check_recency()` |
| Recency: mark older version superseded | ✅ | `datalake.py` → `mark_superseded()` |
| Uniqueness: exact duplicate via hash | ✅ | `verify.py` → `_check_uniqueness()` sha256 hash |
| Uniqueness: near-duplicate via embedding similarity | ✅ | Cosine similarity with `NEAR_DUPLICATE_THRESHOLD` |
| Configurable similarity thresholds | ✅ | `NEAR_DUPLICATE_THRESHOLD` env var in ConfigMap |
| Pass/fail + reason per check | ✅ | `CheckResult` dataclass in `models.py` |
| Failed docs moved to quarantine | ✅ | `quarantine.py` → `quarantine_document()` |
| Failed docs not chunked/embedded/indexed | ✅ | `run.py` skips quarantined docs |
| Verification report per ingestion run | ✅ | `verify.py` → `generate_report()` |

### A2. Incremental Updates

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Re-running unchanged corpus embeds zero new chunks | ✅ | `run.py` hash comparison via data lake |
| Changing one doc replaces only that doc's chunks | ✅ | `store.py` → `_delete_in_batches()` for orphan chunks |
| Deleting a doc removes orphan chunks | ✅ | `datalake.py` → `find_orphan_chunk_ids()` |
| Replacing with newer version archives older | ✅ | `archive.py` → `archive_document()` |
| Superseded versions excluded from retrieval | ✅ | Chroma metadata filter `superseded=False` |

### A3. Data Lake

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Registry at `/data_lake/registry.json` | ✅ | `datalake.py` → `load_registry()` / `save_registry()` |
| Record per document with metadata | ✅ | `DataLakeRecord` dataclass |
| Chunk IDs tracked per document | ✅ | `DataLakeRecord.chunk_ids` |
| PVC provisioned for data lake | ✅ | `k8s/base/datalake-pvc.yaml` |

### A4. Quarantine

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Quarantine directory at `/quarantine` | ✅ | `quarantine.py` uses `QUARANTINE_DIR` env |
| Sidecar JSON with reason | ✅ | `quarantine_document()` writes `<doc>.quarantine.json` |
| PVC provisioned for quarantine | ✅ | `k8s/base/quarantine-pvc.yaml` |

### A5. Archive

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Archive directory at `/archive` | ✅ | `archive.py` uses `ARCHIVE_DIR` env |
| Superseded docs archived with metadata | ✅ | `archive_document()` copies + writes metadata JSON |
| PVC provisioned for archive | ✅ | `k8s/base/archive-pvc.yaml` |

### A6. Corpus Batch B

| Requirement | Status | Evidence |
|-------------|--------|----------|
| ≥ 10 documents with planted problems | ✅ | `corpus/batch_b/` – 10 documents |
| Exact duplicate planted | ✅ | `academic-rules-2026-copy.md`, `fees-2026-duplicate.md` |
| Near-duplicate planted | ✅ | `academic-rules-2026-paraphrased.md` |
| Missing metadata planted | ✅ | `scholarship-policy-no-metadata.txt` |
| Truncated document planted | ✅ | `exam-policy-truncated.txt` |
| Newer version planted | ✅ | `academic-rules-2027.md` (supersedes 2026) |
| Valid new documents | ✅ | `hostel-rules-2026.md`, `library-policy-2026.txt`, `transport-routes-2026.md` |
| Injected instruction planted | ✅ | `it-policy-with-injection.md` |

---

## Part B – Evaluation Suite

### B1. Embedding Evaluation

| Requirement | Status | Evidence |
|-------------|--------|----------|
| ≥ 15 paraphrase pairs | ✅ | `eval/embed_eval.py` – 15 pairs |
| ≥ 15 unrelated pairs | ✅ | `eval/embed_eval.py` – 15 pairs |
| Paraphrase similarity > unrelated similarity | ✅ | Assert in embed_eval.py |
| Timing measurement | ✅ | Per-query embed time recorded |
| Determinism check | ✅ | Same text → same vector check |

### B2. Retrieval Evaluation

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Recall@3 and Recall@5 | ✅ | `eval/retrieval_eval.py` |
| MRR | ✅ | `eval/retrieval_eval.py` |
| nDCG@5 | ✅ | `eval/retrieval_eval.py` |
| Noise/typo robustness test | ✅ | `retrieval_eval.py` injects typos |
| Loads from `eval/golden.jsonl` | ✅ | Reads JSONL fixture |

### B3. End-to-End Evaluation

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Faithfulness judge (rule-based token overlap) | ✅ | `eval/e2e_eval.py` |
| 8 manual labels for judge agreement | ✅ | `e2e_eval.py` – manual_labels dict |
| Abstention rate | ✅ | `e2e_eval.py` |
| Citation validity | ✅ | `e2e_eval.py` |
| p50 and p95 latency | ✅ | `e2e_eval.py` |

### B4. Stochasticity

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Runs eval twice, compares metrics | ✅ | `eval/stochasticity_eval.py` |
| Reports variance across runs | ✅ | `stochasticity_eval.py` |

### B5. One-command runner

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Single command runs all evals | ✅ | `eval/run_all.py` |
| Output saved to `eval/results/` | ✅ | `--out` flag |

---

## Part C – Change Experiment & Release Gate

### C1. Change Experiment

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Variable changed: chunk_size 400 → 200 | ✅ | `eval/config/baseline_config.yaml` vs `candidate_config.yaml` |
| Candidate collection: `rahbar-candidate` | ✅ | `candidate-deployment.yaml` env |
| Candidate indexed separately | ✅ | K8s manifests + `k8s/README.md` step 9 |

### C2. Release Gate

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Configurable thresholds | ✅ | `eval/config/thresholds.yaml` |
| Recall@5 max drop | ✅ | threshold: 0.05 |
| Faithfulness minimum | ✅ | threshold: 0.60 |
| p95 latency maximum | ✅ | threshold: 10.0 s |
| Exit code 0 = PASS, 1 = FAIL | ✅ | `eval/release_gate.py` |
| Kubernetes Eval Job | ✅ | `k8s/base/eval-job.yaml` |
| Thresholds as ConfigMap | ✅ | `k8s/base/eval-thresholds-configmap.yaml` |

---

## Part D – Query Tracing

### D1. Per-query trace records

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Trace ID | ✅ | `api/app/services/trace.py` – UUID |
| Timestamp | ✅ | ISO-8601 in trace record |
| Original query | ✅ | Stored in trace |
| Retrieved passages + scores | ✅ | Stored in trace |
| Retrieval latency | ✅ | Measured in `answer.py` |
| Generation latency | ✅ | Measured in `answer.py` |
| Cache hit/miss | ✅ | Stored in trace |
| Abstain flag | ✅ | Stored in trace |
| Guardrail injection flag | ✅ | Stored in trace |
| Written to `/traces/traces.jsonl` | ✅ | `trace.py` → `write_trace()` |
| PVC provisioned for traces | ✅ | `k8s/base/traces-pvc.yaml` |

### D2. Trace inspect CLI

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Find trace by ID | ✅ | `scripts/trace_inspect.py --id` |
| Pretty-print full pipeline view | ✅ | Formatted output |

### D3. Trace summary

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Abstention rate over time window | ✅ | `scripts/trace_summary.py` |
| Cache hit rate | ✅ | `trace_summary.py` |
| p95 latency | ✅ | `trace_summary.py` |
| Injection flag count | ✅ | `trace_summary.py` |

### D4. Guardrail logging

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Injection detection on query | ✅ | `answer.py` → `_detect_injection()` – 5 patterns |
| Injection detection on retrieved text | ✅ | `answer.py` checks retrieved passages |
| Logged in trace record | ✅ | `guardrail_triggered` field in trace |

---

## Part E – Live Query Coverage

### E1. Live queries

| Requirement | Status | Evidence |
|-------------|--------|----------|
| ≥ 40 live queries | ✅ | `eval/live_queries.jsonl` – 40 queries |
| Mix of in-domain and out-of-domain | ✅ | 28 in-domain + 12 out-of-domain |

### E2. Coverage checker

| Requirement | Status | Evidence |
|-------------|--------|----------|
| BOW cosine similarity vs golden set | ✅ | `eval/coverage.py` |
| Configurable threshold | ✅ | `--threshold` flag; default 0.65 |
| Exit code for below-threshold | ✅ | `coverage.py` returns non-zero |

### E3. Drift experiment

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Coverage drops when out-of-domain queries added | ✅ | Documented in `eval/live_queries.jsonl` |
| Threshold chosen at 0.65 | ✅ | `eval/config/thresholds.yaml` + coverage.py default |

---

## Part F – Kubernetes Demonstrations

| Requirement | Status | Evidence |
|-------------|--------|----------|
| PVC for data lake | ✅ | `k8s/base/datalake-pvc.yaml` |
| PVC for quarantine | ✅ | `k8s/base/quarantine-pvc.yaml` |
| PVC for archive | ✅ | `k8s/base/archive-pvc.yaml` |
| PVC for traces | ✅ | `k8s/base/traces-pvc.yaml` |
| Indexer Job with all volume mounts | ✅ | `k8s/base/indexer-job.yaml` updated |
| Reindex CronJob with all volume mounts | ✅ | `k8s/base/reindex-cronjob.yaml` updated |
| API Deployment with traces volume mount | ✅ | `k8s/base/api-deployment.yaml` updated |
| Eval Kubernetes Job | ✅ | `k8s/base/eval-job.yaml` |
| Candidate Deployment (chunk_size=200) | ✅ | `k8s/base/candidate-deployment.yaml` |
| Candidate Service | ✅ | `k8s/base/candidate-service.yaml` |
| Eval thresholds ConfigMap | ✅ | `k8s/base/eval-thresholds-configmap.yaml` |
| A03 env vars in ConfigMap | ✅ | `k8s/base/configmap.yaml` updated |
| kustomization.yaml includes all new resources | ✅ | `k8s/base/kustomization.yaml` updated |
| Full deploy guide | ✅ | `k8s/README.md` |

**Screenshots required from live cluster (STOP points):**
- [ ] `kubectl get all,pvc,configmap,cronjob,job -n rahbar` (Step 4)
- [ ] Indexer Job logs – original corpus ingestion (Step 5)
- [ ] Indexer Job logs – second run showing zero new chunks (Step 5)
- [ ] Indexer Job logs – Batch B ingestion showing quarantine (Step 6)
- [ ] Eval Job logs – gate PASS result (Step 8)
- [ ] Eval Job logs – gate FAIL demo (Step 9)
- [ ] Candidate deployment running (Step 9)
- [ ] `kubectl rollout undo` output (Step 10)

---

## Part G – Report

| Requirement | Status | Evidence |
|-------------|--------|----------|
| 8-page PDF report | ⬜ PENDING | Requires real experimental results (manual) |
| System architecture diagram | ⬜ PENDING | After K8s demo |
| Evaluation results table | ⬜ PENDING | After `eval/run_all.py` runs |
| Change experiment comparison | ⬜ PENDING | After candidate eval |
| Verification report analysis | ⬜ PENDING | After Batch B ingestion |
| Group member contribution statement | ⬜ PENDING | Requires Abdul-Mueed92 commits |

---

## CI/CD Status

| Check | Status | Notes |
|-------|--------|-------|
| Lint (ruff) | ✅ Passes | As of `ab68c4a` on `dev` |
| Type check (mypy) | ✅ Passes | Fixed in `ab68c4a` |
| Unit tests | ⚠️ Needs verify | Pre-existing issue; check after push |
| Coverage ≥ 60% | ⚠️ Needs verify | `--cov-fail-under=60` in ci.yml |
| Trivy security scan | ⚠️ Set `exit-code: "1"` | Currently `"0"` – will not fail on HIGH/CRITICAL |

---

## Group Participation

| Author | Target | Current |
|--------|--------|---------|
| hassanahsan843-pixel | ≥ 65% | Check with `git shortlog -sn` |
| Abdul-Mueed92 | ≥ 35% | Needs commits before submission |

> **Note:** Abdul-Mueed92 must commit at least 35% of commits. Run:
> ```bash
> git shortlog -sn --all
> ```
> to verify before final submission.
