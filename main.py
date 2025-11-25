"""Simple demo: run preprocessing/normalization over sample texts."""

from __future__ import annotations

import numpy as np

from backend.llm_infrastructure.preprocessing.normalize_engine import (
    build_normalizer,
    sanitize_variant_map,
)
from backend.llm_infrastructure.embedding.registry import get_embedder
from backend.config.settings import rag_settings


def main() -> None:
    sample_texts = [
        "PM 2-1 alarm (1234) helium leak 4.0x10^-9 mt 5",
        "ball screw at pm 2-1 with ll 1",
        "Temp 100 °C pres 50 psi",
    ]

    # 예시 변형어 맵: 사전에 안전 필터를 적용해 사용
    raw_variants = {"exhasut": "exhaust", "pm": "PMX"}
    variant_map = sanitize_variant_map(raw_variants)

    # L0, L3, L4, L5 예시 정규화 실행
    levels = ["L0", "L3", "L4", "L5"]
    normalizers = {lvl: build_normalizer(lvl, variant_map) for lvl in levels}

    for text in sample_texts:
        print(f"\n[Original] {text}")
        for lvl in levels:
            print(f"  [{lvl}] {normalizers[lvl](text)}")

    # --- Embedding demo (registry + alias 기반) ---
    print("\n=== Embedding Demo ===")
    # 레지스트리 alias: bge_base/multilingual_e5/koe5
    embedder = get_embedder(
        rag_settings.embedding_method,
        version=rag_settings.embedding_version,
        device=rag_settings.embedding_device,
        use_cache=rag_settings.embedding_use_cache,
        cache_dir=rag_settings.embedding_cache_dir,
    )

    text = "pm 2-1 helium leak"
    vec = embedder.embed(text)
    vecs = embedder.embed_batch(["hello world", "embedding test"])

    print(f"Embedding (single) shape: {vec.shape}, norm={float(np.linalg.norm(vec)):.4f}")
    print(f"Embedding (batch) shape: {vecs.shape}")


if __name__ == "__main__":
    main()
