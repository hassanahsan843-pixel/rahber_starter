# RAHBAR – Kubernetes Deployment Guide (A03 RAGOps)

This guide walks through every `kubectl` command needed to deploy and demonstrate
the full RAGOps pipeline on a local Minikube (or any K8s cluster).

## Which overlay?

| Overlay | Use | LLM / embeddings | Corpus |
|---------|-----|------------------|--------|
| `k8s/base` = `k8s/overlays/dev` | Offline CI (KinD in `ci.yml`) | `LLM_PROVIDER=stub`, `OLLAMA_URL=offline` (deterministic vectors) | 3 synthetic docs from the `rahbar-corpus` ConfigMap |
| `k8s/overlays/local` | **Local demo without a Groq key** (kind or Minikube) | `LLM_PROVIDER=ollama-host`, `llama3.2:3b` and `nomic-embed-text` from Ollama on the host (`host.docker.internal`) | Synthetic ConfigMap corpus; Batch B from the `rahbar-corpus-batch-b` ConfigMap at `/corpus_b` |
| `k8s/overlays/prod` | **Minikube demo / real deployment** | `LLM_PROVIDER=hosted` (Groq, key from the Secret), Ollama on the Minikube host for `nomic-embed-text` | Real corpus on the `rahbar-corpus` PVC |

`k8s/base/release-gate-job.yaml` re-runs only the release gate on the saved eval results
with the current `eval-thresholds` ConfigMap (apply it on demand, after the eval Job).

Do not change `k8s/base/configmap.yaml` for the demo: CI depends on its offline values.
The real API key only ever lives in the `rahbar-secret` Secret, never in a ConfigMap or file.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Docker | ≥ 24 |
| Minikube | ≥ 1.33 (with Calico, so NetworkPolicies are enforced) |
| kubectl | ≥ 1.29 |
| Ollama on the host | `ollama pull nomic-embed-text` |

```bash
minikube start --driver=docker --cni=calico --cpus=4 --memory=6144
minikube addons enable ingress
minikube addons enable metrics-server
```

---

## Step 1 – Build and load the images

The manifests use five images: baseline (`:dev`), candidate (`:a03`) and eval.

```bash
docker build -t rahbar-api:dev     ./api
docker build -t rahbar-indexer:dev ./indexer
docker build -t rahbar-ui:dev      ./ui
docker build -t rahbar-eval:dev    -f eval/Dockerfile .
docker tag rahbar-api:dev     rahbar-api:a03
docker tag rahbar-indexer:dev rahbar-indexer:a03

for img in rahbar-api:dev rahbar-api:a03 rahbar-indexer:dev rahbar-indexer:a03 rahbar-ui:dev rahbar-eval:dev; do
  minikube image load "$img"
done
```

---

## Step 2 – Stage the corpus on the PVC

```bash
python corpus/fetch.py && python corpus/fetch.py --verify   # downloads corpus/raw/

kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/overlays/prod/corpus-pvc.yaml
kubectl apply -f scripts/corpus-loader-pod.yaml
kubectl wait --for=condition=Ready pod/rahbar-corpus-loader -n rahbar --timeout=120s
kubectl cp corpus/raw/. rahbar/rahbar-corpus-loader:/corpus
# Batch B for the verification demo (Step 6) goes in a sub-folder of the same PVC:
kubectl exec -n rahbar rahbar-corpus-loader -- mkdir -p /corpus/batch_b
kubectl cp corpus/batch_b/. rahbar/rahbar-corpus-loader:/corpus/batch_b
kubectl delete pod rahbar-corpus-loader -n rahbar --wait=true
```

---

## Step 3 – Create the Secret

```bash
kubectl create secret generic rahbar-secret \
  --namespace rahbar \
  --from-literal=LLM_API_KEY=<your-groq-api-key> \
  --from-literal=CACHE_INVALIDATION_TOKEN="$(openssl rand -hex 32)" \
  --dry-run=client -o yaml | kubectl apply -f -
```

`CACHE_INVALIDATION_TOKEN` is required: the API Deployments and every indexer Job read
it from this Secret. The indexer sends it to `POST /api/cache/invalidate` after indexing;
without it, pods fail with `CreateContainerConfigError`. If you rotate the token,
restart the API Deployments (`kubectl rollout restart deploy -n rahbar`) so they pick it up.

---

## Step 4 – Apply the manifests and wait for the index

```bash
kubectl apply -k k8s/overlays/prod
kubectl get all,pvc,configmap,cronjob,job -n rahbar      # screenshot for Part F
kubectl wait --for=condition=complete job/rahbar-indexer -n rahbar --timeout=900s
kubectl logs -n rahbar -l app=rahbar-indexer --tail=50
```

Expected: 5 PVCs (`datalake-pvc`, `quarantine-pvc`, `archive-pvc`, `traces-pvc`, `rahbar-corpus`),
the indexer log ends with `api cache invalidation completed` and `indexer finished`.
The first invalidation attempt may wait a few seconds for the API pods to turn Ready
(the indexer retries for ~24 s).

Re-run to show incremental indexing (expected: `Zero new chunks embedded – corpus unchanged`):

```bash
kubectl create job --from=cronjob/reindex reindex-check -n rahbar
kubectl logs -n rahbar job/reindex-check --follow
```

---

## Step 5 – Verify the API

```bash
kubectl port-forward svc/rahbar-api 8000:8000 -n rahbar &
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
curl -X POST http://localhost:8000/api/ask \
     -H 'Content-Type: application/json' \
     -d '{"question":"What CGPA does an undergraduate student need to stay off academic warning?"}'
```

---

## Step 6 – Batch B ingestion (verification demo)

Batch B is ingested **alongside** the existing corpus. Every registry record belongs to
a batch (here `batch_b`, from the folder name), and a run only removes documents of its
own batch, so the Batch A / production documents stay indexed.

```bash
kubectl create job --from=cronjob/reindex batch-b-1 -n rahbar --dry-run=client -o json \
  | python -c "import json,sys; j=json.load(sys.stdin); j['spec']['template']['spec']['containers'][0]['args']=['--corpus-dir=/corpus/batch_b','--batch=batch_b']; print(json.dumps(j))" \
  | kubectl apply -f -
kubectl logs -n rahbar job/batch-b-1 --follow
```

Expected log lines: `quarantined` for `exam-policy-truncated.txt` (truncation) and
`scholarship-policy-no-metadata.txt` (no date/version); `recency_family` reporting
`academic-rules-2027.md` as a newer version of `academic-rules-2026.md`; and
`verification_report_written` with the summary. The full report is written to
`/data_lake/reports/verification_report_<run_id>.json` on `datalake-pvc`.

---

## Step 7 – Evaluation Job (B1–B5 + release gate)

```bash
kubectl logs -n rahbar job/rahbar-eval --follow
```

The Job (in the prod overlay it scores with `eval/golden.yaml`, the golden set for the real corpus):
1. Runs `eval/run_all.py` against the baseline API → `/traces/baseline_results.json`
2. Runs it against the candidate API → `/traces/candidate_results.json`
3. Runs `eval/release_gate.py` → `/traces/gate_result_latest.json`, exit 0 (PASS) or 1 (FAIL)

To re-run it: `kubectl delete job rahbar-eval -n rahbar && kubectl apply -k k8s/overlays/prod`.

---

## Step 8 – Chunk-size experiment (Part C)

The candidate API (`rahbar-api-candidate`, collection `rahbar-candidate`, chunk 200/30) is
created by Step 4. Build its index from the **same** corpus PVC as the baseline:

```bash
kubectl apply -f k8s/overlays/prod/candidate-indexer-job.yaml
kubectl wait --for=condition=complete job/rahbar-candidate-indexer -n rahbar --timeout=900s
kubectl delete job rahbar-eval -n rahbar && kubectl apply -k k8s/overlays/prod   # re-run Step 7
```

Use the prod candidate Job, not `k8s/base/candidate-indexer-job.yaml`: the base one reads the
synthetic ConfigMap corpus, where every document is shorter than 200 words, so chunk sizes
400 and 200 produce identical indexes and the experiment would compare nothing.

Run the gate locally on copied results if needed:

```bash
python eval/release_gate.py \
  --baseline traces/baseline_results.json \
  --candidate traces/candidate_results.json \
  --thresholds eval/config/thresholds.yaml \
  --out traces/gate_result.json
```

---

## Step 9 – Rollout / Rollback Demo (Part F)

```bash
# Promote candidate → production
kubectl set image deployment/rahbar-api api=rahbar-api:a03 -n rahbar
kubectl rollout status deployment/rahbar-api -n rahbar

# Rollback if gate fails
kubectl rollout undo deployment/rahbar-api -n rahbar
kubectl rollout status deployment/rahbar-api -n rahbar
```

---

## Step 10 – Trace Inspection (Part D)

```bash
# Copy traces from pod to local
kubectl cp -n rahbar \
  $(kubectl get pod -n rahbar -l app=rahbar-api -o jsonpath='{.items[0].metadata.name}'):/traces/traces.jsonl \
  traces/traces.jsonl

# Inspect a specific trace (the trace id is the request id)
python scripts/trace_inspect.py <trace-id> --file traces/traces.jsonl

# Summarize all traces
python scripts/trace_summary.py --file traces/traces.jsonl
```

---

## Step 11 – Coverage Check (Part E)

```bash
python eval/coverage.py \
  --live eval/live_queries.jsonl \
  --golden eval/golden.jsonl \
  --threshold 0.65
```

Coverage compares queries with the pinned embedding model through Ollama
(`OLLAMA_URL`, default `http://localhost:11434`; `EMBED_MODEL`, default
`nomic-embed-text`). Without Ollama it exits 2; `--method bow` runs the older
bag-of-words comparison offline, which misses paraphrases.

---

## Teardown

```bash
kubectl delete -k k8s/overlays/prod
minikube stop
```
