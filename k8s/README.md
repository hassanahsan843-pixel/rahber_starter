# RAHBAR – Kubernetes Deployment Guide (A03 RAGOps)

This guide walks through every `kubectl` command needed to deploy and demonstrate
the full RAGOps pipeline on a local Minikube (or any K8s cluster).

---

## Prerequisites

| Tool | Version |
|------|---------|
| Docker | ≥ 24 |
| Minikube | ≥ 1.33 |
| kubectl | ≥ 1.29 |
| kustomize | bundled with kubectl |

```bash
minikube start --cpus=4 --memory=6144
eval $(minikube docker-env)   # on Linux/macOS
# Windows PowerShell:
# minikube docker-env | Invoke-Expression
```

---

## Step 1 – Build Images

```bash
docker build -t rahbar-api:dev    ./api
docker build -t rahbar-indexer:dev ./indexer
docker build -t rahbar-ui:dev     ./ui
```

---

## Step 2 – Create the Secret

```bash
kubectl create secret generic rahbar-secret \
  --namespace rahbar \
  --from-literal=LLM_API_KEY=<your-groq-api-key> \
  --dry-run=client -o yaml | kubectl apply -f -
```

---

## Step 3 – Apply the Full Manifests

```bash
kubectl apply -k k8s/base
```

Expected output includes:
- namespace/rahbar configured
- 4 PersistentVolumeClaims created (datalake-pvc, quarantine-pvc, archive-pvc, traces-pvc)
- configmap/rahbar-config configured
- All Deployments, Services, Jobs, CronJobs, Ingress

---

## Step 4 – Verify All Resources

```bash
kubectl get all,pvc,configmap,cronjob,job -n rahbar
```

Screenshot this output for Part F evidence.

---

## Step 5 – Run the Indexer Job (Original Corpus)

```bash
# Watch the job until completion
kubectl logs -n rahbar -l app=rahbar-indexer --follow

# After completion, verify zero-new-chunk behaviour on second run:
kubectl delete job rahbar-indexer -n rahbar
kubectl apply -f k8s/base/indexer-job.yaml
kubectl logs -n rahbar -l app=rahbar-indexer --follow
```

Expected: second run prints `Zero new chunks embedded – corpus unchanged.`

---

## Step 6 – Run Batch B Ingestion (Verification Demo)

The Batch B corpus is in `corpus/batch_b/`. To demonstrate the verification pipeline,
temporarily update `CORPUS_DIR` in the ConfigMap or pass `--corpus-dir` as a Job arg:

```bash
kubectl delete job rahbar-indexer -n rahbar
# Edit k8s/base/indexer-job.yaml to add: args: ["--corpus-dir=/corpus/batch_b"]
kubectl apply -f k8s/base/indexer-job.yaml
kubectl logs -n rahbar -l app=rahbar-indexer --follow
```

Expected log output includes:
- Quarantined documents with reasons (missing metadata, truncated, duplicate)
- Verification report summary
- Archival of superseded version

---

## Step 7 – Verify API Readiness

```bash
kubectl port-forward svc/rahbar-api 8000:8000 -n rahbar &
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
curl -X POST http://localhost:8000/query \
     -H 'Content-Type: application/json' \
     -d '{"question":"What is the minimum CGPA to avoid academic probation?"}'
```

---

## Step 8 – Run the Evaluation Job

```bash
kubectl apply -f k8s/base/eval-job.yaml
kubectl logs -n rahbar -l app=rahbar-eval --follow
```

The Job:
1. Runs `eval/run_all.py` against the live API → saves results to `/traces/eval_results_latest.json`
2. Runs `eval/release_gate.py` → saves gate result to `/traces/gate_result_latest.json`
3. Exits 0 (PASS) or 1 (FAIL)

---

## Step 9 – Change Experiment (Part C1)

Deploy the candidate (chunk_size=200, collection=rahbar-candidate):

```bash
# 1. Index corpus into candidate collection
kubectl delete job rahbar-indexer -n rahbar
# Edit indexer-job.yaml to set CHROMA_COLLECTION=rahbar-candidate and CHUNK_SIZE=200
kubectl apply -f k8s/base/indexer-job.yaml
kubectl logs -n rahbar -l app=rahbar-indexer --follow

# 2. Deploy candidate API
kubectl apply -f k8s/base/candidate-deployment.yaml
kubectl apply -f k8s/base/candidate-service.yaml

# 3. Run eval against candidate
kubectl port-forward svc/rahbar-api-candidate 8001:8000 -n rahbar &
python eval/run_all.py --api http://localhost:8001 --out eval/results/candidate_results.json
python eval/release_gate.py \
  --baseline eval/config/baseline_config.yaml \
  --results eval/results/candidate_results.json \
  --thresholds eval/config/thresholds.yaml \
  --out eval/results/gate_candidate.json
```

---

## Step 10 – Rollout / Rollback Demo (Part F)

```bash
# Promote candidate → production
kubectl set image deployment/rahbar-api api=rahbar-api:dev -n rahbar
kubectl rollout status deployment/rahbar-api -n rahbar

# Rollback if gate fails
kubectl rollout undo deployment/rahbar-api -n rahbar
kubectl rollout status deployment/rahbar-api -n rahbar
```

---

## Step 11 – Trace Inspection (Part D)

```bash
# Copy traces from pod to local
kubectl cp -n rahbar \
  $(kubectl get pod -n rahbar -l app=rahbar-api -o jsonpath='{.items[0].metadata.name}'):/traces/traces.jsonl \
  traces/traces.jsonl

# Inspect a specific trace
python scripts/trace_inspect.py --id <trace-id> --file traces/traces.jsonl

# Summarize all traces
python scripts/trace_summary.py --file traces/traces.jsonl
```

---

## Step 12 – Coverage Check (Part E)

```bash
python eval/coverage.py \
  --live eval/live_queries.jsonl \
  --golden eval/golden.jsonl \
  --threshold 0.65
```

---

## Teardown

```bash
kubectl delete -k k8s/base
minikube stop
```
