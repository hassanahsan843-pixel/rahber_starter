# ADR 0003: Redis as the Persistent Semantic Cache Store

## Status
Accepted · 2026-11-05

## Context
Repeated, semantically identical natural language queries ("What is the passing CGPA?" vs "What CGPA is required to pass?") incur significant generation latency (1-3 seconds) and LLM inference cost. To optimize query serving, section 3.3 requires a semantic cache that computes cosine similarity between incoming query embeddings and cached question representations.

We must choose the storage and indexing tier for the semantic cache between two architectural options:
1. **A dedicated Redis instance with persistent storage (Deployment + PVC)**.
2. **A secondary ChromaDB vector collection**.

The cache system must support:
- Strict cosine similarity thresholding (read dynamically from `CACHE_THRESHOLD` in ConfigMap).
- High-throughput vector lookups and low sub-millisecond key-value retrieval.
- Atomic eviction and complete invalidation upon corpus re-indexing.
- Robust exclusion rules: never caching follow-ups, honest abstentions (`answered_from_context: false`), out-of-scope refusals, or validation fallback errors.

## Decision
We select **Redis with an AOF (Append-Only File) persistent volume claim (PVC)** as the semantic cache storage engine.

Query embeddings and their associated schema-constrained answers are stored in Redis. When a question arrives:
1. Follow-up and malformed queries bypass the cache check entirely.
2. The question is embedded using the pinned embedding model (`EMBED_MODEL`).
3. Cosine similarity is computed against cached query vectors.
4. If similarity $\ge$ `CACHE_THRESHOLD` (tuned to `0.90`), the validated JSON response is returned immediately with cache metadata (`cached: true`).
5. Upon completion of an indexing Job or CronJob, a cache invalidation signal (`FLUSHDB` or generation epoch bump) is executed to guarantee that old, superseded answers are never served over a refreshed index.

## Options Considered

### Option 1: Redis with Persistent Volume (Selected)
- **Advantages**:
  - Redis operates in-memory with sub-millisecond latencies, significantly outpacing ChromaDB query round-trips.
  - Native key expiration (TTL) and atomic database clearing (`FLUSHDB`) ensure immediate and fail-safe cache invalidation when a re-index completes.
  - Decouples retrieval metadata operations from vector search operations, preventing cache churn from locking or polluting the primary document vector index.
  - Minimal container footprint (~30MB RAM idle).
- **Trade-offs**: Requires deploying a dedicated `Deployment`, `Service`, and `PersistentVolumeClaim` in Kubernetes.

### Option 2: Secondary ChromaDB Collection
- **Advantages**: Reuses existing ChromaDB StatefulSet infrastructure without provisioning an additional pod.
- **Disadvantages**:
  - ChromaDB is optimized for batch document indexing and nearest-neighbor search, not fast, dynamic point-in-time cache evictions or TTLs.
  - Deleting or flushing an entire Chroma collection locks internal index files and degrades query latency on the primary collection.
  - Chroma telemetry and SQLite locking overhead introduce unnecessary latency for simple cache lookups.

## Consequences
### Positive
- Sub-5ms cache hit latency compared to 1500ms+ full generation latency.
- Clean invalidation boundary: When the indexer Job finishes, running `redis-cli FLUSHDB` prevents serving obsolete answers.
- Zero risk of polluting the primary academic document collection with conversational query text.

### Negative & Mitigations
- Additional resource requirements: Allocated small, bounded limits (`cpu: 200m`, `memory: 128Mi`) and 500Mi PVC, which has negligible overhead on Minikube.
- NetworkPolicy must explicitly authorize API-to-Redis communication on port 6379 while strictly blocking any unauthorized external or UI access.
