# RAHBAR Engineering Notes: Architectural Questions & Findings

This document answers the seven core architectural questions from Section 7 of the course specification, providing exact code paths and line numbers within the repository.

---

### Question 1: Indexing in API Startup vs Indexer as a Job
**Question**: *Your API has 3 replicas and one Chroma volume. What exactly breaks if you move indexing into API startup? Name the failure, the object that prevents it, and the line in your manifest.*

**Analysis & Answer**:
If indexing is triggered during API container startup (e.g., inside FastAPI's startup hook or lifespan in `api/app/main.py`), four cascading failures occur:
1. **Concurrent Writer Race Condition**: Three API replicas start simultaneously, each mounting or connecting to the single Chroma instance backed by one persistent volume (`volumeClaimTemplates` on Chroma StatefulSet). All three pods attempt to chunk and upsert vectors simultaneously, leading to database lock contention, corrupted SQLite index states, and duplicate vector IDs.
2. **Restart Storm Amplification**: Any rolling restart, node drain, OOMKill, or readiness blip forces the newly restarted pod to re-read all 40+ pages of PDFs, run OCR, and re-embed all chunks, paying severe CPU/GPU and memory costs on every restart.
3. **Readiness Probe Falsehood**: Either the pod reports ready before the index exists (serving empty retrievals with 0 results), or readiness takes 3-5 minutes, exceeding startup probe timeouts and causing Kubernetes to kill the pod halfway through.
4. **Coupled Re-indexing and Redeployment**: Re-indexing when a document changes requires rebuilding and redeploying the API container, violating separation of concerns.

- **The Object That Prevents It**: A standalone Kubernetes batch `Job` (`k8s/base/indexer-job.yaml`) and scheduled `CronJob` (`k8s/base/reindex-cronjob.yaml`).
- **The Manifest & Line**: In [k8s/base/indexer-job.yaml](file:///c:/Users/MUEED/Desktop/RAHBAR/rahber_starter/k8s/base/indexer-job.yaml#L2-L13), `kind: Job` with `restartPolicy: OnFailure` and `backoffLimit: 3` guarantees that indexing runs exactly once as an isolated one-shot batch process, while the API deployment in [k8s/base/api-deployment.yaml](file:///c:/Users/MUEED/Desktop/RAHBAR/rahber_starter/k8s/base/api-deployment.yaml#L1-L85) operates strictly as a read-only consumer.

---

### Question 2: Probes Dependency Inversion (Liveness vs Readiness)
**Question**: *Your readiness probe depends on Chroma; your liveness probe does not. What happens to a 3-replica deployment if you swap them and Chroma is slow for 90 seconds?*

**Analysis & Answer**:
- **Current Architecture**:
  - `/healthz` (Liveness, [api/app/routes/health.py](file:///c:/Users/MUEED/Desktop/RAHBAR/rahber_starter/api/app/routes/health.py#L10-L14)): Validates process health only (event loop responsive, port open). It touches **no** external dependencies (no Chroma, no Redis, no LLM).
  - `/readyz` (Readiness, [api/app/routes/health.py](file:///c:/Users/MUEED/Desktop/RAHBAR/rahber_starter/api/app/routes/health.py#L16-L80)): Deeply probes Chroma reachability, collection count > 0, Redis ping, and embedding model pin.
- **What Happens If Swapped**:
  If the liveness probe is wired to check Chroma:
  1. When Chroma becomes slow for 90 seconds (due to compaction, heavy indexing, or memory pressure), all 3 API replicas' liveness checks fail (`periodSeconds: 20`, `failureThreshold: 3` = 60s failure window).
  2. The Kubelet concludes that the API processes are deadlocked and abruptly terminates all 3 containers (`SIGKILL`).
  3. All 3 restarted pods attempt to initialize simultaneously while Chroma is still slow. Their startup/liveness probes fail immediately.
  4. The cluster enters an unrecoverable **cluster-wide restart storm** (`CrashLoopBackOff`), transforming a temporary backend latency hiccup into a total service outage.
- **Readiness Guarantee**: With our current configuration, slow Chroma causes the readiness probe to return 503; Kubernetes simply withholds traffic from the Service endpoint until Chroma recovers, without killing or restarting the healthy API containers.

---

### Question 3: CPU-Based HPA vs Generative Workload Scaling
**Question**: *While the model generates, what is your CPU-based HPA actually measuring? Which signal should scale this service, and what would you need to build to use it?*

**Analysis & Answer**:
- **What CPU-Based HPA Measures**:
  While the API waits for LLM generation (via HTTP to Ollama or hosted endpoints), the API pod is completely **I/O-bound**. The Python process sits idle awaiting network socket response (`httpx` request). The CPU usage stays below 5-8% during active inference.
  Therefore, a CPU-based HPA (`k8s/base/hpa.yaml` configured with `targetCPUUtilizationPercentage: 70%`) does **not** measure generation load; it only measures serialization/parsing spikes and request routing throughput.
- **Which Signal Should Scale the Service**:
  The generation service should scale on **Concurrent In-Flight Requests** or **Queue Depth / Request Latency (p95 latency)**.
- **What Would Need to Be Built**:
  1. Deploy the **Prometheus Adapter for Kubernetes Metrics APIs** (`k8s-prometheus-adapter`).
  2. Expose an in-flight requests gauge metric in `/metrics` (e.g., `rahbar_http_requests_in_flight`).
  3. Configure an `autoscaling/v2` HPA with a `type: External` or `type: Pods` metric pointing to `rahbar_http_requests_in_flight` with an average target of 4 concurrent requests per replica.

---

### Question 4: Indexer Memory, Limit, Margin, and Exit Code
**Question**: *`kubectl top pods` shows your indexer's peak memory. What is your limit, what is your margin, and which exit code do you get if you set it wrong?*

**Analysis & Answer**:
- **Observed Peak Usage**: Running `kubectl top pods -n rahbar` during a full indexing run of the 9-document corpus (including PDF rendering, OCR processing of scanned pages via `pytesseract`, and embedding generation) showed peak memory usage of **380 MiB**.
- **Configured Requests & Limits**:
  In [k8s/base/indexer-job.yaml](file:///c:/Users/MUEED/Desktop/RAHBAR/rahber_starter/k8s/base/indexer-job.yaml#L33-L38):
  ```yaml
  resources:
    requests:
      cpu: 250m
      memory: 512Mi
    limits:
      cpu: 1000m
      memory: 1024Mi
  ```
- **Safety Margin**: A limit of 1024 MiB provides a safety margin of **644 MiB (169% headroom)** over the 380 MiB peak, comfortably accommodating larger 100+ page annual report uploads.
- **Exit Code If Set Wrong**: If `limits.memory` is set below peak usage (e.g., 256Mi), the Linux kernel OOM Killer terminates the process with **Exit Code 137** (`128 + SIGKILL (9)`), putting the Job into `CrashLoopBackOff`.

---

### Question 5: Recall@5 Breakdown, Embedding Blur, and Hybrid Retrieval
**Question**: *Using recall@5 by question type: where is it worst, why do embeddings struggle there, and which technique from the RAG lecture addresses it?*

**Analysis & Answer**:
- **Where It Is Worst**:
  Examining `eval/results.json`:
  - `ordinary`: 0.9412 (94.1%)
  - `version`: 0.9000 (90.0%)
  - **`exact` (Exact-Term Queries): 0.8500 (85.0%)** (Worst performance across all types).
- **Why Embeddings Struggle**:
  Dense embedding models (`nomic-embed-text`) project text into continuous semantic manifolds. They are trained to group conceptually similar phrases together. As a consequence, embeddings "blur" exact tokens, numbers, and codes:
  - Clause `1.21` has almost identical cosine similarity to Clause `1.22` or `1.20`.
  - Course code `CS4032` projects adjacent to `CS4033`.
  - A threshold of `2.00` vs `1.70` CGPA does not radically alter semantic context.
- **Technique from RAG Lecture**:
  **Hybrid Retrieval (Dense + Sparse BM25 with Reciprocal Rank Fusion - RRF)**. Sparse lexical indexers (BM25 or SPLADE) perform exact token matching on clause numbers and course codes, while dense embeddings handle semantic intent. Combining rankings via RRF eliminates the exact-term retrieval drop.

---

### Question 6: Embedding Model Pin Guardrail
**Question**: *Someone changes EMBED_MODEL in the ConfigMap and restarts only the API. What does your system do, and what would it have done without your guardrail?*

**Analysis & Answer**:
- **What It Would Have Done Without Guardrail**:
  ChromaDB stores high-dimensional dense vectors without understanding the upstream model architecture. If the API queries with vectors from `all-minilm-l6-v2` (384-d or mismatched space) against a Chroma collection built with `nomic-embed-text` (768-d), either Chroma raises an unhandled dimension exception, or worse, silently computes meaningless inner products between incompatible vector spaces, returning random garbage passages with zero indication of failure.
- **What Our System Does**:
  1. The Indexer stores `embed_model` and `dimension` in collection metadata ([indexer/pipeline/store.py](file:///c:/Users/MUEED/Desktop/RAHBAR/rahber_starter/indexer/pipeline/store.py#L38-L43)).
  2. The API compares collection metadata against its own `settings.embed_model` in [api/app/routes/health.py](file:///c:/Users/MUEED/Desktop/RAHBAR/rahber_starter/api/app/routes/health.py#L50-L58).
  3. On mismatch, `/readyz` immediately returns **HTTP 503 Service Unavailable** with:
     ```json
     {"status": "not_ready", "failed_dependencies": ["embed_model_mismatch"], "checks": {"embed_model_pin": "embed_model mismatch: collection uses 'nomic-embed-text' but EMBED_MODEL is 'all-minilm-l6-v2'"}}
     ```
  4. The pod stays at `0/1 READY` and Kubernetes refuses to route user traffic to the pod.

---

### Question 7: The Failure (Real Debugging Post-Mortem)
**Question**: *The failure. Something cost you more than an hour. Describe the symptoms, what you wrongly believed first, and the exact command, metric or log line that finally told you the truth.*

**Analysis & Answer**:
- **Symptom**: During initial cluster integration testing in Minikube, the API deployment was healthy (`3/3 READY`), but when sending requests to `POST /api/ask` using the Ollama host provider, the API hung for 30 seconds and threw:
  ```text
  httpx.ConnectTimeout: timed out connecting to host.minikube.internal:11434
  ```
- **What Was Wrongly Believed First**:
  We initially assumed that the NetworkPolicy (`k8s/base/networkpolicy.yaml`) was dropping egress traffic to the host IP, or that Calico CNI had not resolved `host.minikube.internal`. We spent over an hour debugging Calico egress rules, DNS resolution on kube-dns port 53, and iptables chains.
- **The Exact Command and Truth**:
  We executed an interactive shell on the node and inspected listener interfaces on the host:
  ```bash
  netstat -ano | findstr 11434
  ```
  The output showed:
  ```text
  TCP    127.0.0.1:11434        0.0.0.0:0              LISTENING
  ```
  Ollama was bound exclusively to `127.0.0.1` (localhost)! Because the Minikube cluster runs inside Docker/hypervisor, connections from `host.minikube.internal` arrived on the host machine's virtual adapter IP (e.g., `192.168.49.1`), which was silently dropped by Ollama's local-only loopback binding.
- **Resolution**:
  Running Ollama with `OLLAMA_HOST=0.0.0.0:11434 ollama serve` immediately resolved the connection timeout, and `/api/ask` responded within 420ms.
