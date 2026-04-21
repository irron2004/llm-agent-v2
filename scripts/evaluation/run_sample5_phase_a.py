"""Run the sample5 eval queries against Phase A API to produce after-Phase-A evidence.

5 queries are taken from `data/eval_results/mybot_march_react_after_20260416_sample5/run_min.jsonl`
so before/after comparison is 1:1 consistent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
BEFORE_FILE = REPO / "data/eval_results/mybot_march_react_before_20260416_sample5/run_min.jsonl"
DEFAULT_OUT = REPO / "data/eval_results/mybot_march_react_phase_a_20260421_sample5"
DEFAULT_API = "http://localhost:8611/api/agent/run"
TIMEOUT = 900  # seconds per query (Qwen3 with think=true may take longer for large refs)


def load_before_queries() -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    for line in BEFORE_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        queries.append((rec["qid"], rec["query"]))
    return queries


def run_query(qid: str, query: str, api_url: str) -> dict:
    payload = {
        "message": query,
        "use_react_agent": True,
        "max_attempts": 2,
        "auto_parse": True,
    }
    t0 = time.time()
    try:
        r = requests.post(api_url, json=payload, timeout=TIMEOUT)
        elapsed = (time.time() - t0) * 1000
        r.raise_for_status()
        resp = r.json()
    except Exception as exc:
        elapsed = (time.time() - t0) * 1000
        return {"qid": qid, "query": query, "elapsed_ms": elapsed, "error": str(exc)}
    # Match schema of previous runs
    minimal = {
        "judge": resp.get("judge"),
        "interrupt_payload": resp.get("interrupt_payload"),
        "search_queries": resp.get("search_queries"),
        "detected_language": resp.get("detected_language"),
        "metadata": resp.get("metadata"),
        "query": resp.get("query") or query,
        "answer": resp.get("answer"),
        "selected_doc_types": resp.get("selected_doc_types"),
        "interrupted": resp.get("interrupted"),
        "thread_id": resp.get("thread_id"),
    }
    return {"qid": qid, "query": query, "elapsed_ms": elapsed, "response": minimal}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=DEFAULT_API)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run_min.jsonl"
    queries = load_before_queries()
    print(f"Running {len(queries)} queries against {args.api_url}")
    results: list[dict] = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, (qid, q) in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {qid}: {q[:80]}", flush=True)
            result = run_query(qid, q, args.api_url)
            faith = (result.get("response") or {}).get("judge", {}).get("faithful", "?")
            ans_len = len((result.get("response") or {}).get("answer", "") or "")
            elapsed = result.get("elapsed_ms", 0)
            err = result.get("error", "")
            print(f"    → faithful={faith} answer_len={ans_len} elapsed={elapsed:.0f}ms {err}", flush=True)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results.append(result)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
