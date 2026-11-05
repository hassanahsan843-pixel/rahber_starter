# ADR 0001: Decoupling Document Indexing from API Runtime as a Batch Kubernetes Job

## Status
Accepted · 2026-11-05

## Context
In a Retrieval-Augmented Generation (RAG) architecture deployed on Kubernetes, an obvious but catastrophic anti-pattern is executing corpus extraction, chunking, and embedding during API pod startup (e.g., inside FastAPI `lifespan` or startup hooks). 

Our production specification dictates that the API deployment scales horizontally to at least 3 replicas, backed by a persistent ChromaDB instance over a shared volume. Placing indexing inside the API startup creates four fatal failure modes:

1. **Volume Write Races and State Corruption**: Three API pods booting concurrently would attempt simultaneous writes to a single Chroma PVC. Concurrent chunk writes across uncoordinated processes produce a fractured, unrepeatable vector index representing no coherent state of the underlying corpus.
2. **Restart Multiplication of Compute Cost**: Every pod lifecycle event—a rolling update, an OOMKill, a node drain, or a transient probe blip—triggers a complete re-extraction and re-embedding of the multi-document corpus. This wastes GPU/CPU cycles and saturates embedding providers.
3. **Readiness Probe Invalidation**: If the pod marks itself ready before indexing completes, it serves empty retrievals and hallucinated responses. Conversely, if readiness waits for indexing (taking minutes for large PDFs), Kubernetes startup and liveness probes time out and kill the pod, leading to a cascading CrashLoopBackOff where no working pod remains to serve user requests.
4. **Configuration Lock-In (The Friday Failure)**: Ingesting new documents or re-indexing cannot be performed without redeploying the entire API service, directly violating zero-downtime operations.

The fundamental operational principle from our lecture is: **indexing runs once, and again when material changes; querying runs on every question**.

## Decision
We decouple document indexing completely from the API service process:
- **Writer Path**: Document extraction, cleaning, chunking, embedding, and database mutation run strictly within an ephemeral, one-shot Kubernetes **`Job`** (`rahbar-indexer`) and a scheduled **`CronJob`** (`reindex`). The Job runs under `restartPolicy: OnFailure` with a bounded `backoffLimit: 3`.
- **Reader Path**: The API deployment runs 3 replicas that have **read-only** access to the ChromaDB vector collection. The API's `/readyz` probe validates that the vector collection exists and contains chunks before reporting ready.
- **Zero-Downtime Re-indexing**: Re-indexing is triggered on-demand via a single command:
  ```bash
  kubectl create job --from=cronjob/reindex manual-reindex-1 -n rahbar
  ```
  The API continues serving incoming user traffic uninterrupted while the batch Job computes new vectors.

## Options Considered
1. **Indexing in API `lifespan` startup hook**: Rejected due to concurrency write collisions across replicas, excessive cold-start latencies, and total service disruption during rollouts.
2. **Kubernetes `initContainers` on the API Pod**: Rejected because `initContainers` still execute on *every* replica pod creation and restart, re-running redundant embeddings and failing to decouple index lifecycle from web server lifecycle.
3. **Standalone Asynchronous Background Worker Queue (Celery/RabbitMQ)**: Over-engineered for scheduled/manual corpus updates, adding unnecessary operational dependencies to the cluster.
4. **Dedicated Kubernetes Job and CronJob (Selected)**: Aligns natively with cloud-native workload primitives, guarantees run-to-completion semantics, enforces single-writer exclusivity, and enables isolated resource limits.

## Consequences
### Positive
- Guarantees zero write-contention on ChromaDB storage.
- Rapid, predictable API pod startup (< 2 seconds), enabling responsive horizontal autoscaling (HPA) from 3 to 8 replicas.
- API deployments and rollouts can occur with `maxUnavailable: 0` without risking stale or missing index states.
- Re-indexing runs as an isolated batch workload with dedicated CPU/memory limits without impacting query serving latencies.

### Negative & Mitigations
- Vector index staleness until the batch Job completes: Mitigated by scheduling automated weekly CronJob updates and supporting instant manual job triggers.
- Cache invalidation coordination: Mitigated by triggering Redis semantic cache flushes upon successful Indexer Job completion.
