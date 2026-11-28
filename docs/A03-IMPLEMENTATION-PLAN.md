# Assignment 03 – RAGOps: Implementation Plan

**Project:** RAHBAR (Retrieval-Augmented Handbook-Based Academic Resource)  
**Branch:** `a03-dev` (branched from `dev` / `assignment-02-final` tag)  
**Authors:** Hassan Ahsan (hassanahsan843-pixel), Abdul Mueed (Abdul-Mueed92)  
**Date:** 2026-11-21

---

## 1. Existing A02 Implementation Summary

### What exists and works

| Component | Location | Status |
|-----------|----------|--------|
| FastAPI query API | `api/app/` | ✅ Working – routes, schema, middleware |
| Chroma retriever | `api/app/retrieval/chroma.py` | ✅ Implemented; fallback offline embedding |
| Stub retriever | `api/app/retrieval/stub.py` | ✅ Used in unit tests |
| Hosted LLM provider (Groq) | `api/app/providers/llm/hosted.py` | ✅ Implemented |
| Stub LLM provider | `api/app/providers/llm/stub.py` | ✅ Used in unit tests |
| Redis semantic cache | `api/app/services/cache.py` | ✅ Distributed lookup + invalidation |
| Prometheus metrics | `api/app/routes/metrics.py` | ✅ Retrieval/generation latency, cache hits, abstentions |
| JSON structured logging | `api/app/logjson.py` | ✅ Working |
| Document extraction | `indexer/pipeline/extract.py` | ✅ PDF/MD/HTML, OCR fallback |
| Document cleaning | `indexer/pipeline/clean.py` | ✅ Noise removal |
| Chunking (structure-aware) | `indexer/pipeline/chunk.py` | ✅ Section-aware, overlap, stable IDs |
| Embedding (Ollama / fallback) | `indexer/pipeline/embed.py` | ✅ Offline deterministic fallback |
| Chroma store (upsert) | `indexer/pipeline/store.py` | ✅ Upsert with model metadata |
| Indexer entry point | `indexer/pipeline/run.py` | ✅ Corpus discovery, invalidation |
| Kubernetes manifests | `k8s/base/` | ✅ Full set: API, Chroma, Redis, HPA, Ingress |
| CI pipeline | `.github/workflows/ci.yml` | ✅ Lint, type, test, build, e2e-eval |
| CD pipeline | `.github/workflows/cd.yml` | ✅ Smoke test, Ingress check |
| Evaluation script | `eval/run_eval.py` | ✅ Recall@3/5, citation validity, abstention gate |
| Fixture golden set | `eval/fixtures/ci.golden.yaml` | ✅ 6 questions incl. must-abstain |
| Corpus (3 docs) | `indexer/corpus/` | ✅ academic-rules-2026.md, fees-2026.md, academic-rules-2023.md |
| Submission checker | `scripts/check_submission.py` | ✅ 166 passed, 0 failures |

### Known Problems / Gaps for A03

| # | Problem | Impact |
|---|---------|--------|
| P1 | No document **verification** stage (quality, completeness, recency, uniqueness checks) | Missing A2 Part A1 |
| P2 | No **incremental indexing** – store.py deletes entire collection on every run | Missing A2 Part A2 |
| P3 | No **data lake** management (persistent records of ingested documents) | Missing F1 PVC |
| P4 | No **quarantine** directory handling in pipeline | Missing A2 Part A1 |
| P5 | No **document versioning / archival** beyond `superseded` flag in Chroma metadata | Missing A2 Part A2 |
| P6 | No **Batch B** – only 3 clean corpus documents | Missing A2 Part A3 |
| P7 | No **golden.jsonl** (25-question set) for A3 evaluation | Missing B |
| P8 | No **embedding evaluation** (paraphrase similarity, determinism test) | Missing B1 |
| P9 | No **MRR, nDCG@5** metrics in eval script | Missing B2 |
| P10 | No **faithfulness judge**, latency p50/p95 in eval | Missing B3 |
| P11 | No **stochasticity test** (run twice, compare metrics) | Missing B4 |
| P12 | No **one-command eval runner** (`eval/run_all.py`) | Missing B5 |
| P13 | No **change experiment** (baseline vs. candidate config) | Missing C1 |
| P14 | No **release gate config file** (`eval/config/thresholds.yaml`) | Missing C2 |
| P15 | No **candidate Kubernetes Deployment** | Missing C3 / F3 |
| P16 | No per-query **trace records** (JSON lines with all pipeline fields) | Missing D1 |
| P17 | No **trace inspection CLI** | Missing D2 |
| P18 | No **trace summary script** | Missing D3 |
| P19 | No **guardrail logging** in traces | Missing D4 |
| P20 | No **live query log** (40 simulated queries) | Missing E1 |
| P21 | No **coverage checker** script | Missing E2 |
| P22 | No **data drift** experiment | Missing E3 |
| P23 | No **Persistent Volume Claims** for data_lake, quarantine, archive, traces | Missing F1 |
| P24 | No **Evaluation Job** (`k8s/eval-job.yaml`) | Missing F2 |
| P25 | No **compliance checklist** | Missing Phase 11 |

---

## 2. Files to Create (New)

### Indexer / Ingestion
- `indexer/pipeline/verify.py` – A1: document verification (quality, completeness, recency, uniqueness)
- `indexer/pipeline/incremental.py` – A2: incremental update logic, diff against data lake
- `indexer/pipeline/datalake.py` – A2: data lake JSON store (document registry)
- `indexer/pipeline/archive.py` – A2: archive old versions to `archive/`
- `indexer/pipeline/quarantine.py` – A1: move failed docs to `quarantine/`

### Corpus / Batch B
- `corpus/batch_b/` – 10 documents with planted problems (A3)

### Evaluation
- `eval/golden.jsonl` – 25-question golden set (B)
- `eval/embed_eval.py` – B1: embedding module-level tests
- `eval/retrieval_eval.py` – B2: retrieval component tests (MRR, nDCG, noise)
- `eval/e2e_eval.py` – B3: faithfulness, latency p50/p95
- `eval/stochasticity_eval.py` – B4: run twice, compare
- `eval/run_all.py` – B5: one command for entire suite
- `eval/release_gate.py` – C2: load thresholds, compare, exit 0/1
- `eval/coverage.py` – E2: live query coverage checker
- `eval/config/thresholds.yaml` – C2: gate thresholds
- `eval/config/baseline_config.yaml` – C1: baseline pipeline config
- `eval/config/candidate_config.yaml` – C1: candidate pipeline config (changed chunk size)
- `eval/results/` – timestamped result files

### Tracing
- `api/app/services/trace.py` – D1: query trace record writer
- `scripts/trace_inspect.py` – D2: CLI tool to inspect a trace by ID
- `scripts/trace_summary.py` – D3: metrics summary over time window
- `traces/` – persistent trace storage

### Kubernetes
- `k8s/base/eval-job.yaml` – F2: evaluation Job
- `k8s/base/candidate-deployment.yaml` – C3/F3: candidate API Deployment
- `k8s/base/candidate-service.yaml` – C3/F3: candidate Service
- `k8s/base/datalake-pvc.yaml` – F1: PVC for data lake
- `k8s/base/quarantine-pvc.yaml` – F1: PVC for quarantine
- `k8s/base/archive-pvc.yaml` – F1: PVC for archive
- `k8s/base/traces-pvc.yaml` – F1: PVC for traces
- `k8s/README.md` – F5: deploy commands

### Live Queries
- `eval/live_queries.jsonl` – E1: 40 simulated user queries

### Documentation
- `docs/A03-IMPLEMENTATION-PLAN.md` ← this file
- `docs/A03-COMPLIANCE-CHECKLIST.md` – Phase 11

---

## 3. Files to Modify (Existing)

| File | Change |
|------|--------|
| `indexer/pipeline/run.py` | Add verification, incremental logic, data lake, quarantine |
| `indexer/pipeline/store.py` | Change from full-delete to incremental upsert/delete |
| `indexer/pipeline/models.py` | Add `doc_id`, `doc_hash`, `title`, `date`, `page_count` to Document |
| `api/app/services/answer.py` | Write JSON trace record on every query |
| `api/app/routes/ask.py` | Pass trace_id; return trace location |
| `api/app/logjson.py` | Extend to also write traces/ file |
| `k8s/base/kustomization.yaml` | Add new manifest files |
| `k8s/base/indexer-job.yaml` | Add data_lake, quarantine, archive volume mounts |
| `k8s/overlays/dev/kustomization.yaml` | Include new resources |
| `eval/run_eval.py` | Add MRR, nDCG@5 metrics; timestamps |
| `README.md` | Add A–F command sections |

---

## 4. Implementation Order

```
Phase 0  → Audit (DONE – this document)
Phase A  → Verification, Batch B, Incremental indexing, Data lake
Phase B  → golden.jsonl (25 Q), embed eval, retrieval eval, E2E eval, stochasticity, run_all.py
Phase C  → Change experiment, release gate config + script, candidate K8s deployment
Phase D  → Trace records, trace inspect CLI, trace summary, guardrail logging
Phase E  → Live queries, coverage checker, drift experiment
Phase F  → PVCs, eval-job.yaml, candidate deployment, rollout demo
Phase G  → Report (manual – after screenshots and experiments)
Phase 11 → Compliance checklist
```

---

## 5. Git Strategy

- Working branch: `a03-dev`
- Tag at A02 state: `assignment-02-final`
- Checkpoint branch: `checkpoint/assignment-02-final`
- Each phase gets atomic commits under appropriate author
- Never commit real credentials

---

## 6. Known Technical Decisions

1. **Embedding model stays `nomic-embed-text` via Ollama** (or offline fallback). Candidate change experiment will use **chunk size** (400→200) as the controlled variable — simplest to test reliably without a running LLM.
2. **Data lake** = JSON file at `/data_lake/registry.json` inside the pod, backed by a PVC. Tracks doc_id, doc_hash, version, chunk_ids per document.
3. **Incremental logic**: hash each document file → compare against registry → skip unchanged, upsert changed, delete orphans.
4. **Traces** = JSON lines appended to `/traces/traces.jsonl` (PVC) + stdout.
5. **Faithfulness judge** = rule-based for offline reliability (checks whether answer text is a substring/paraphrase of retrieved passages); 8 answers manually labelled for comparison.
