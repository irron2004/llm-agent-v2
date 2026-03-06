from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="chunk_v3 embedding artifact checker")
    parser.add_argument("--embeddings", required=True, help=".npy 파일 경로")
    parser.add_argument("--chunk-ids", required=True, help="chunk_ids JSONL 경로")
    parser.add_argument(
        "--expected-dim", type=int, default=None, help="기대 임베딩 차원"
    )
    parser.add_argument("--check-norm", action="store_true", help="L2 norm 분포도 검사")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embeddings_path = Path(args.embeddings)
    ids_path = Path(args.chunk_ids)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"embeddings not found: {embeddings_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"chunk ids not found: {ids_path}")

    vectors = np.load(str(embeddings_path))
    if vectors.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={vectors.shape}")

    ids: list[str] = []
    with ids_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(str(row["chunk_id"]))

    if vectors.shape[0] != len(ids):
        raise ValueError(
            f"row mismatch: embeddings={vectors.shape[0]} chunk_ids={len(ids)}"
        )

    if args.expected_dim is not None and vectors.shape[1] != args.expected_dim:
        raise ValueError(
            f"dimension mismatch: actual={vectors.shape[1]} expected={args.expected_dim}"
        )

    if np.isnan(vectors).any() or np.isinf(vectors).any():
        raise ValueError("NaN or Inf detected in embeddings")

    if args.check_norm:
        norms = np.linalg.norm(vectors, axis=1)
        print(
            f"norm(min/avg/max)=({norms.min():.6f}/{norms.mean():.6f}/{norms.max():.6f})"
        )

    print(
        f"OK rows={vectors.shape[0]} dims={vectors.shape[1]} ids={len(ids)} file={embeddings_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
