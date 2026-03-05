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


# 임베딩 모델 설정
# dims: ES 인덱스에 저장할 최종 벡터 차원
# native_dims: 모델 원래 차원 (MRL truncation 시 참고)
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen3_emb_4b": {
        "hf_name": "Qwen/Qwen3-Embedding-4B",
        "dims": 1024,
        "native_dims": 2560,
        "truncate_dim": 1024,
        "normalize": "l2",
        "trust_remote_code": False,
        "query_prefix": "query: ",
        "document_prefix": "",
    },
    "bge_m3": {
        "hf_name": "BAAI/bge-m3",
        "dims": 1024,
        "native_dims": 1024,
        "truncate_dim": None,
        "normalize": "l2",
        "trust_remote_code": False,
        "query_prefix": "",
        "document_prefix": "",
    },
    "jina_v5": {
        "hf_name": "jinaai/jina-embeddings-v5-text-small",
        "dims": 1024,
        "native_dims": 1024,
        "truncate_dim": None,
        "normalize": "l2",
        "trust_remote_code": True,
        "query_prefix": "",
        "document_prefix": "",
    },
}


def _create_embedder(model_key: str, device: str = "cuda"):
    """모델별 SentenceTransformerEmbedder 생성."""
    from backend.llm_infrastructure.embedding.engines.sentence.embedder import (
        SentenceTransformerEmbedder,
    )

    cfg = MODEL_CONFIGS[model_key]
    return SentenceTransformerEmbedder(
        model_name=cfg["hf_name"],
        device=device,
        trust_remote_code=cfg.get("trust_remote_code", False),
        truncate_dim=cfg.get("truncate_dim"),
    )


def validate_model_contract(model_key: str, embedder) -> None:
    """모델 contract 검증: dims, NaN, norm."""
    cfg = MODEL_CONFIGS[model_key]
    test_texts = ["hello world", "임베딩 테스트 문장입니다"]
    vecs = embedder.encode(test_texts)

    actual_dims = vecs.shape[1]
    expected_dims = cfg["dims"]
    if actual_dims != expected_dims:
        raise ValueError(
            f"[{model_key}] dims mismatch: expected={expected_dims}, actual={actual_dims}"
        )

    if np.any(np.isnan(vecs)) or np.any(np.isinf(vecs)):
        raise ValueError(f"[{model_key}] NaN/Inf detected in embeddings")

    norms = np.linalg.norm(vecs, axis=1)
    if cfg["normalize"] == "l2" and (norms.min() < 0.95 or norms.max() > 1.05):
        raise ValueError(
            f"[{model_key}] L2 norm out of range: [{norms.min():.4f}, {norms.max():.4f}]"
        )

    print(f"  Contract OK: dims={actual_dims}, norm=[{norms.min():.4f}, {norms.max():.4f}]")


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
    embedder = _create_embedder(model_key, device=device)

    print(f"  Validating model contract...")
    validate_model_contract(model_key, embedder)

    all_vectors: list[np.ndarray] = []
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        vectors = embedder.embed_batch(batch, batch_size=batch_size)
        all_vectors.append(vectors)
        print(f"    Batch {i // batch_size + 1}/{total_batches} done ({len(batch)} texts)")

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
