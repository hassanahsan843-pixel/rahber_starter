# Part F evidence – Kubernetes run (2026-11-28)

Real command output from a local **kind** cluster (`kind-rahbar-a03`, Kubernetes v1.37,
Docker Desktop), deployed with `kubectl apply -k k8s/overlays/local` (Ollama on the host
for `llama3.2:3b` and `nomic-embed-text`). Each file shows the command (`$ …`) followed by
its unedited output. Secret values were never printed (verified: neither the cache token
nor the LLM key value appears in any file here).

| File | What it shows |
|---|---|
| `00-cluster.txt` | context and node |
| `01-secret-and-configmaps.txt` | Secret exists (2 keys, values not shown); Batch B ConfigMap |
| `02-apply.txt` | `kubectl apply -k k8s/overlays/local` |
| `03-ingestion-job.txt` | ingestion Job completed (24 chunks, cache invalidated) |
| `04-rollout.txt` | API rollout status (candidate waits for its collection, see 08) |
| `05-resources.txt` | `kubectl get all,pvc,configmap,cronjob,job -n rahbar`; rollout strategy; requests/limits |
| `06-batch-b-ingestion.txt`, `06-batch-b-verification-report.txt` | Batch B Job; verification report read from `datalake-pvc` |
| `07-unchanged-rerun.txt` | unchanged re-run: zero chunks embedded |
| `08-candidate-index.txt`, `10-candidate-v2-batch-b.txt` | independent `rahbar-candidate` collection |
| `09-evaluation-job.txt` | eval Job: candidate v1 (Batch A only) rejected by the gate |
| `11-evaluation-job-candidate-v2.txt` | eval Job: candidate v2 rejected on abstention (0.846 < 1.0) |
| `12-gate-configmap-change.txt` | gate Job: FAIL → ConfigMap edit → PASS → original restored → FAIL |
| `13-rollout-rollback.txt` | rollout (`maxUnavailable: 0`), status, history, `rollout undo`, trace before/after |
| `14-metrics-and-traces.txt` | `/metrics` via port-forward; trace in pod logs; `traces.jsonl` on `traces-pvc` |
| `15-trace-inspect-and-summary.txt` | `trace_inspect.py` (prompt hash verified) and `trace_summary.py` over 1,651 traces |

Notes, so nothing is over-read:
- The ingestion Job's first container attempt failed (Ollama busy with a parallel local run);
  the Job's retry completed it. `newly_embedded: 0` in its log means it reused the
  document vectors cached by that first attempt.
- The eval Job has `backoffLimit: 1`; after a failing gate the first pod's logs were kept
  and the Job deleted so it would not re-run the 30-minute suite. Its results stay on
  `traces-pvc`, which the gate Job (`k8s/base/release-gate-job.yaml`) reads.
- In `12-…` the value `abstention_correctness_min: 0.8` is a configurability demonstration
  only; the committed release criterion remains 1.0 and was restored in the same run.
- The synthetic corpus is too small for chunk size to matter (see `corpus/batch_b/README.md`
  and `eval/results/partc_*`, where the 400 vs 200 comparison runs on the real corpus).
- Screenshots for the report must be taken from your own terminal; these text files are
  the underlying output.
