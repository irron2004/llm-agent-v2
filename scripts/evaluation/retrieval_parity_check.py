from __future__ import annotations

import argparse
import http.client
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast


@dataclass
class ParityResult:
    matched: bool
    mismatch_index: int | None
    agent_value: str | None
    retrieval_value: str | None
    missing_from_agent: list[str]
    missing_from_retrieval: list[str]


@dataclass(frozen=True)
class CliArgs:
    base_url: str
    query: str
    k: int
    deterministic: bool
    agent_timeout: float
    retrieval_timeout: float


def _post_json(
    url: str, payload: Mapping[str, object], timeout: float
) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with cast(
            http.client.HTTPResponse, urllib.request.urlopen(req, timeout=timeout)
        ) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    try:
        parsed = cast(object, json.loads(body.decode("utf-8")))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected JSON shape from {url}: expected object")
    parsed_map = cast(dict[object, object], parsed)
    normalized: dict[str, object] = {}
    for key, value in parsed_map.items():
        normalized[str(key)] = value
    return normalized


def _coerce_str_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw_map = cast(dict[object, object], value)
    result: dict[str, object] = {}
    for key, item in raw_map.items():
        result[str(key)] = item
    return result


def _extract_agent_doc_ids(agent_response: Mapping[str, object]) -> list[str]:
    docs = agent_response.get("retrieved_docs")
    if not isinstance(docs, list):
        raise RuntimeError("Agent response missing 'retrieved_docs' list")
    ids: list[str] = []
    for doc in cast(list[object], docs):
        doc_map = _coerce_str_mapping(doc)
        if doc_map is None:
            continue
        value = doc_map.get("id")
        if isinstance(value, str):
            ids.append(value.strip())
        elif value is not None:
            ids.append(str(value).strip())
    return ids


def _extract_retrieval_doc_ids(retrieval_response: Mapping[str, object]) -> list[str]:
    docs = retrieval_response.get("docs")
    if not isinstance(docs, list):
        raise RuntimeError("Retrieval response missing 'docs' list")
    ids: list[str] = []
    for doc in cast(list[object], docs):
        doc_map = _coerce_str_mapping(doc)
        if doc_map is None:
            continue
        value = doc_map.get("doc_id")
        if isinstance(value, str):
            ids.append(value.strip())
        elif value is not None:
            ids.append(str(value).strip())
    return ids


def _compare_ordered(
    agent_ids: list[str], retrieval_ids: list[str], k: int
) -> ParityResult:
    agent_topk = agent_ids[:k]
    retrieval_topk = retrieval_ids[:k]

    mismatch_index: int | None = None
    agent_value: str | None = None
    retrieval_value: str | None = None

    max_len = max(len(agent_topk), len(retrieval_topk))
    for idx in range(max_len):
        left = agent_topk[idx] if idx < len(agent_topk) else None
        right = retrieval_topk[idx] if idx < len(retrieval_topk) else None
        if left != right:
            mismatch_index = idx
            agent_value = left
            retrieval_value = right
            break

    agent_set = set(agent_topk)
    retrieval_set = set(retrieval_topk)
    missing_from_agent = [
        doc_id for doc_id in retrieval_topk if doc_id not in agent_set
    ]
    missing_from_retrieval = [
        doc_id for doc_id in agent_topk if doc_id not in retrieval_set
    ]

    return ParityResult(
        matched=agent_topk == retrieval_topk,
        mismatch_index=mismatch_index,
        agent_value=agent_value,
        retrieval_value=retrieval_value,
        missing_from_agent=missing_from_agent,
        missing_from_retrieval=missing_from_retrieval,
    )


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Check retrieval parity between /api/agent/run and /api/retrieval/run"
    )
    _ = parser.add_argument(
        "--base-url", default="http://localhost:8001", help="API base URL"
    )
    _ = parser.add_argument(
        "--query", required=True, help="Query to compare across endpoints"
    )
    _ = parser.add_argument(
        "--k", type=int, default=10, help="Compare first K ranked doc IDs"
    )
    _ = parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set deterministic mode for /api/retrieval/run (default: true)",
    )
    _ = parser.add_argument(
        "--agent-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for /api/agent/run",
    )
    _ = parser.add_argument(
        "--retrieval-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for /api/retrieval/run",
    )
    parsed = parser.parse_args()

    base_url_raw = getattr(parsed, "base_url", "")
    query_raw = getattr(parsed, "query", "")
    k_raw = getattr(parsed, "k", 10)
    deterministic_raw = getattr(parsed, "deterministic", True)
    agent_timeout_raw = getattr(parsed, "agent_timeout", 30.0)
    retrieval_timeout_raw = getattr(parsed, "retrieval_timeout", 30.0)

    return CliArgs(
        base_url=str(base_url_raw),
        query=str(query_raw),
        k=int(k_raw),
        deterministic=bool(deterministic_raw),
        agent_timeout=float(agent_timeout_raw),
        retrieval_timeout=float(retrieval_timeout_raw),
    )


def main() -> int:
    args = _parse_args()
    if args.k <= 0:
        print("FAIL: --k must be >= 1")
        return 1

    base_url = args.base_url.rstrip("/")
    agent_url = f"{base_url}/api/agent/run"
    retrieval_url = f"{base_url}/api/retrieval/run"

    agent_payload = {
        "message": args.query,
        "mode": "base",
        "auto_parse": False,
        "max_attempts": 0,
        "use_canonical_retrieval": True,
        "top_k": args.k,
    }
    retrieval_payload = {
        "query": args.query,
        "steps": ["retrieve"],
        "deterministic": bool(args.deterministic),
    }

    try:
        agent_response = _post_json(
            agent_url, agent_payload, timeout=args.agent_timeout
        )
        retrieval_response = _post_json(
            retrieval_url,
            retrieval_payload,
            timeout=args.retrieval_timeout,
        )
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1

    try:
        agent_ids = _extract_agent_doc_ids(agent_response)
        retrieval_ids = _extract_retrieval_doc_ids(retrieval_response)
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1

    result = _compare_ordered(agent_ids, retrieval_ids, k=args.k)

    print(f"Compared first K={args.k} doc IDs")
    print(f"Agent (/api/agent/run): {agent_ids[: args.k]}")
    print(f"Retrieval (/api/retrieval/run): {retrieval_ids[: args.k]}")

    if result.mismatch_index is None:
        print("First mismatch index: none")
    else:
        print(
            f"First mismatch index: {result.mismatch_index}"
            + f" (agent={result.agent_value!r}, retrieval={result.retrieval_value!r})"
        )

    print(f"Missing from agent list: {result.missing_from_agent}")
    print(f"Missing from retrieval list: {result.missing_from_retrieval}")

    if result.matched:
        print("PASS: ranked doc ID lists match exactly")
        return 0

    print("FAIL: ranked doc ID lists differ")
    return 1


if __name__ == "__main__":
    sys.exit(main())
