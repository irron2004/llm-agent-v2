# Retrieval Quality Analysis Report

**Date:** 2026-02-26  
**Analysis:** Retrieval inconsistency check (same query -> different docs)

---

## Summary

This report documents a current-state reproduction of the retrieval inconsistency bug, where identical queries can return different documents depending on non-deterministic factors in the retrieval pipeline.

---

## Reproduction Details

### Endpoint
```
POST http://localhost:8011/api/retrieval/run
```

### Request Payload
```python
import requests

url = "http://localhost:8011/api/retrieval/run"
payload = {
    "query": "PM setting 유지 절차 알려줘",
    "steps": ["retrieve"],
    "debug": False,
    "auto_parse": False,
    "rerank_enabled": False,
    "deterministic": False  # toggled between True/False
}
response = requests.post(url, json=payload)
```

---

## Observed Results

### Mode: `deterministic=false` (5 runs)

| Run | Top-5 Doc IDs |
|-----|---------------|
| 0 | `['40020644','40090411','40090413','40114268','40153983']` |
| 1 | `['40020644','40114268','40153983','40018311','40068660']` |
| 2 | `['global_sop_precia_all_pm_prevent_maintenance','GCB_33745',...]` |
| 3 | *(different)* |
| 4 | *(different)* |

**Stability:** 5 runs produced **4 unique** top-k doc_id lists.

### Mode: `deterministic=true` (5 runs)

| Run | Top-5 Doc IDs |
|-----|---------------|
| All | `['40138445','40041488','40158571','40094392','40073110']` |

**Stability:** 5 runs produced **1 unique** top-k doc_id list.

---

## Key Observation

In non-deterministic mode, the retrieval results varied not only in ranking order but also in the **document ID scheme** itself:
- Some runs returned purely numeric IDs (e.g., `40020644`)
- Other runs returned prefixed IDs (e.g., `global_sop_precia_all_pm_prevent_maintenance`)

The mixed doc_id schemes suggest the retrieval path may vary across runs in non-deterministic mode, which could indicate nondeterministic retrieval path behavior.

---

## Why This Happens

The behavior is controlled by the `deterministic` flag in the request, which changes the **MQ (Multi-Query) strategy** used during retrieval.

### Request Field
```python
# backend/api/routers/retrieval.py, line 46
class RetrievalRunRequest(BaseModel):
    deterministic: bool = False  # defaults to False
```

### Backend Policy
```python
# backend/services/retrieval_effective_config.py, line 87
"policies": {
    "mq_strategy": "bypass" if deterministic or applied_skip_mq else "llm",
}
```

### Mechanism
- When `deterministic=true` (or `skip_mq=true`): `mq_strategy` is set to `"bypass"`, meaning the original query is used directly without query expansion.
- When `deterministic=false`: `mq_strategy` is set to `"llm"`, meaning an LLM-driven multi-query expansion is applied to generate multiple search queries.

Because the LLM-driven expansion introduces sampling variability (temperature, sampling), each run can produce different `search_queries`, leading to different retrieved documents and different doc_id schemes appearing in results.

### Frontend Default
The frontend harness (`frontend/src/features/retrieval-test/hooks/use-retrieval-test.ts`) defaults to `deterministic: true`, which is why the test harness produces stable results.

### Comparing Search Queries
To verify this mechanism, fetch the run snapshots and compare `search_queries` between runs:
```python
import requests

# Run with deterministic=False (may vary)
payload_false = {
    "query": "PM setting 유지 절차 알려줘",
    "steps": ["retrieve"],
    "debug": False,
    "auto_parse": False,
    "rerank_enabled": False,
    "deterministic": False,
}
r_false = requests.post("http://localhost:8011/api/retrieval/run", json=payload_false)
run_id_false = r_false.json()["run_id"]

# Run with deterministic=True (stable)
payload_true = {
    "query": "PM setting 유지 절차 알려줘",
    "steps": ["retrieve"],
    "debug": False,
    "auto_parse": False,
    "rerank_enabled": False,
    "deterministic": True,
}
r_true = requests.post("http://localhost:8011/api/retrieval/run", json=payload_true)
run_id_true = r_true.json()["run_id"]

# Fetch run snapshots
details_false = requests.get(f"http://localhost:8011/api/retrieval/runs/{run_id_false}").json()
details_true = requests.get(f"http://localhost:8011/api/retrieval/runs/{run_id_true}").json()

# Compare mq_strategy and search_queries
print("deterministic=False:")
print("  mq_strategy:", details_false.get("effective_config", {}).get("policies", {}).get("mq_strategy"))
print("  search_queries:", details_false.get("search_queries"))
print("deterministic=True:")
print("  mq_strategy:", details_true.get("effective_config", {}).get("policies", {}).get("mq_strategy"))
print("  search_queries:", details_true.get("search_queries"))
```

---

## Conclusion

**The issue still occurs.**  
When `deterministic=false`, the same query can return materially different document sets across runs. Setting `deterministic=true` stabilizes the output to a single consistent result.

---

## Recommended Next Steps

1. **Default to `deterministic=true`** for production workflows requiring consistent retrieval results.

2. **Investigate the root cause** of non-deterministic behavior in the retrieval pipeline (likely in the vector store or embedding generation).

3. **Add logging** to identify which retrieval path is taken when different doc_id schemes appear.

4. **Consider a configuration flag** to enforce deterministic retrieval by default, with an explicit opt-out for cases where variety is desired.

---

*This report is focused on the original retrieval inconsistency bug and current-state reproduction. It is separate from other analysis documents.*
