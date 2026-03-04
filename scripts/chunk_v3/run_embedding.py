"""Phase 3: 멀티모델 배치 임베딩 스크립트.

JSONL chunks를 로드하여 모델별로 임베딩하고 .npy로 저장한다.

Usage:
    python scripts/chunk_v3/run_embedding.py \
        --chunks data/chunks_v3/all_chunks.jsonl \
        --models bge_m3 koe5 \
        --output-dir data/chunks_v3/ \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chunk_v3.common import load_chunks_jsonl


# 임베딩 모델 설정 (추후 확정 시 업데이트)
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "bge_m3": {
        "hf_name": "BAAI/bge-m3",
        "dims": 1024,
        "normalize": "l2",
        "query_prefix": "",
        "document_prefix": "",
    },
    "koe5": {
        "hf_name": "nlpai-lab/KoE5",
        "dims": 1024,
        "normalize": "l2",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "multilingual_e5": {
        "hf_name": "intfloat/multilingual-e5-large",
        "dims": 1024,
        "normalize": "l2",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
}


def embed_model(
    model_key: str,
    texts: list[str],
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """단일 모델로 텍스트 배치 임베딩.

    Args:
        model_key: MODEL_CONFIGS 키
        texts: 임베딩할 텍스트 리스트
        batch_size: 배치 크기
        device: 디바이스 (cuda/cpu)

    Returns:
        (N, dims) numpy array
    """
    cfg = MODEL_CONFIGS[model_key]

    from backend.llm_infrastructure.embedding.engines.sentence.embedder import (
        SentenceTransformerEmbedder,
    )

    embedder = SentenceTransformerEmbedder(
        model_name=cfg["hf_name"],
        device=device,
    )

    all_vectors: list[np.ndarray] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        vectors = embedder.embed(batch)
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        all_vectors.append(vectors)
        print(f"    Batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} done")

    return np.vstack(all_vectors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: 멀티모델 배치 임베딩")
    parser.add_argument(
        "--chunks", required=True,
        help="입력 JSONL 파일 경로 (all_chunks.jsonl)",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
        help=f"임베딩 모델 키 목록 (기본: {list(MODEL_CONFIGS.keys())})",
    )
    parser.add_argument(
        "--output-dir", default="data/chunks_v3",
        help="출력 디렉토리 (기본: data/chunks_v3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="배치 크기 (기본: 64)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="디바이스 (기본: cuda)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load chunks
    print(f"Loading chunks from {args.chunks}...")
    chunks = load_chunks_jsonl(args.chunks)
    print(f"  Loaded {len(chunks)} chunks")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract chunk_ids for all models
    chunk_ids = [c.chunk_id for c in chunks]

    for model_key in args.models:
        if model_key not in MODEL_CONFIGS:
            print(f"  WARNING: Unknown model '{model_key}', skipping")
            continue

        cfg = MODEL_CONFIGS[model_key]
        print(f"\nEmbedding with {model_key} ({cfg['hf_name']})...")

        # Apply document prefix
        doc_prefix = cfg.get("document_prefix", "")
        texts = [doc_prefix + c.content for c in chunks]

        # Embed
        vectors = embed_model(
            model_key=model_key,
            texts=texts,
            batch_size=args.batch_size,
            device=args.device,
        )

        # Save embeddings (.npy)
        npy_path = output_dir / f"embeddings_{model_key}.npy"
        np.save(str(npy_path), vectors)
        print(f"  Saved embeddings: {npy_path} (shape: {vectors.shape})")

        # Save chunk_ids (.jsonl)
        ids_path = output_dir / f"chunk_ids_{model_key}.jsonl"
        with open(ids_path, "w", encoding="utf-8") as f:
            for cid in chunk_ids:
                f.write(json.dumps({"chunk_id": cid}, ensure_ascii=False) + "\n")
        print(f"  Saved chunk IDs: {ids_path}")

    print(f"\n{'='*60}")
    print("EMBEDDING COMPLETE")
    print(f"  Models: {args.models}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
