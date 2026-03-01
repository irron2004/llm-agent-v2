#!/usr/bin/env python3
"""Create regression run workspace manifest.

This script creates a JSON manifest that captures metadata for a before/after
regression comparison run, including SHAs, query file stats, ES configuration,
and filtered environment variables.

Usage:
    python scripts/evaluation/regression_compare_manifest.py \
        --run-id "20260101_120000" \
        --before-sha 73ca832 \
        --after-sha c04fa25 \
        --queries-subset ".sisyphus/evidence/paper-b/task-10/queries_subset.jsonl" \
        --queries-full "data/synth_benchmarks/stability_bench_v1/queries.jsonl" \
        --es-host "http://localhost:8002" \
        --es-env synth \
        --es-index-prefix rag_synth \
        --out ".sisyphus/evidence/regression_compare/20260101_120000/manifest.json"
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


# Whitelisted env var prefixes
ENV_PREFIXES = ("SEARCH_", "ES_", "RAG_", "VLLM_")
# Secrets suffixes to redact (case-insensitive)
SECRET_SUFFIXES = ("_KEY", "_PASSWORD", "_TOKEN")


def _redact_value(name: str, value: str) -> str:
    """Redact value if name contains secret suffix."""
    name_lower = name.lower()
    for suffix in SECRET_SUFFIXES:
        if suffix.lower() in name_lower:
            return "***REDACTED***"
    return value


def _compute_file_stats(path: Path) -> dict[str, Any]:
    """Compute sha256 and line count for a file."""
    sha256_hash = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256_hash.update(chunk)

    # Count lines
    with open(path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)

    return {
        "path": str(path),
        "sha256": sha256_hash.hexdigest(),
        "line_count": line_count,
    }


def _get_filtered_env() -> dict[str, str]:
    """Get whitelisted env vars with secrets redacted."""
    filtered: dict[str, str] = {}
    for key, value in os.environ.items():
        # Check if key starts with any whitelisted prefix
        if any(key.startswith(prefix) for prefix in ENV_PREFIXES):
            filtered[key] = _redact_value(key, value)
    return filtered


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create regression run workspace manifest"
    )
    _ = parser.add_argument(
        "--run-id",
        required=True,
        help="Unique run identifier (YYYYMMDD_HHMMSS format)",
    )
    _ = parser.add_argument(
        "--before-sha",
        default="73ca832",
        help="Git SHA for before state (default: 73ca832)",
    )
    _ = parser.add_argument(
        "--after-sha",
        default="c04fa25",
        help="Git SHA for after state (default: c04fa25)",
    )
    _ = parser.add_argument(
        "--queries-subset",
        required=True,
        help="Path to subset query file (JSONL)",
    )
    _ = parser.add_argument(
        "--queries-full",
        required=True,
        help="Path to full query file (JSONL)",
    )
    _ = parser.add_argument(
        "--es-host",
        required=True,
        help="Elasticsearch host URL",
    )
    _ = parser.add_argument(
        "--es-env",
        required=True,
        help="Elasticsearch environment (e.g., synth)",
    )
    _ = parser.add_argument(
        "--es-index-prefix",
        required=True,
        help="Elasticsearch index prefix (e.g., rag_synth)",
    )
    _ = parser.add_argument(
        "--out",
        required=True,
        help="Output path for manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Validate run-id is non-empty
    if not args.run_id or not args.run_id.strip():
        print("ERROR: --run-id cannot be empty", file=sys.stderr)
        return 1

    # Convert paths to Path objects
    queries_subset_path = Path(args.queries_subset)
    queries_full_path = Path(args.queries_full)
    out_path = Path(args.out)

    # Validate query files exist
    if not queries_subset_path.exists():
        print(
            f"ERROR: Query subset file not found: {queries_subset_path}",
            file=sys.stderr,
        )
        return 1

    if not queries_full_path.exists():
        print(f"ERROR: Query full file not found: {queries_full_path}", file=sys.stderr)
        return 1

    # Create parent directories for output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute file stats
    subset_stats = _compute_file_stats(queries_subset_path)
    full_stats = _compute_file_stats(queries_full_path)

    # Get filtered environment variables
    env_vars = _get_filtered_env()

    # Build manifest
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = {
        "run_id": args.run_id,
        "created_at": now,
        "before_sha": args.before_sha,
        "after_sha": args.after_sha,
        "query_files": {
            "subset": subset_stats,
            "full": full_stats,
        },
        "es_config": {
            "host": args.es_host,
            "env": args.es_env,
            "index_prefix": args.es_index_prefix,
        },
        "environment": env_vars,
    }

    # Write manifest with indent=2 and trailing newline
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Manifest created: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
