from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.elasticsearch.mappings import get_chunk_v3_content_mapping
from scripts.chunk_v3.chunkers import chunk_gcb, chunk_myservice, chunk_vlm_parsed
from scripts.chunk_v3.common import canonicalize_doc_type
from scripts.chunk_v3.run_embedding import embed_model
import scripts.chunk_v3.run_embedding as run_embedding_module


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sop", "sop"),
        ("TS", "ts"),
        ("trouble_shooting", "ts"),
        ("troubleshooting", "ts"),
        ("setup_manual", "setup"),
        ("set_up_manual", "setup"),
        ("installation manual", "setup"),
        ("myservice", "myservice"),
        ("gcb", "gcb"),
    ],
)
def test_canonicalize_doc_type_contract(raw: str, expected: str) -> None:
    assert canonicalize_doc_type(raw) == expected


@pytest.mark.parametrize("doc_type_alias", ["setup_manual", "set_up_manual"])
def test_chunk_vlm_parsed_setup_alias_normalizes_to_setup(
    tmp_path: Path, doc_type_alias: str
) -> None:
    fixture = {
        "doc_id": "setup_doc",
        "source_file": "SUPRA_XP_setup_manual.pdf",
        "source_type": "pdf",
        "total_pages": 1,
        "vlm_model": "test-vlm",
        "pages": [
            {
                "page": 1,
                "text": "CHAPTER 1\nSTEP 1\n밸브를 점검한다.\nSTEP 2\n압력을 확인한다.",
            }
        ],
    }
    json_path = tmp_path / "setup.json"
    json_path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")

    chunks = chunk_vlm_parsed(doc_type_alias, json_path)

    assert chunks
    assert all(c.doc_type == "setup" for c in chunks)
    assert all(c.extra_meta.get("source_doc_type") == doc_type_alias for c in chunks)
    sequence = [int(c.extra_meta.get("sequence_no", 0)) for c in chunks]
    assert sequence == list(range(1, len(chunks) + 1))


def test_chunk_myservice_section_aware_multichunk(tmp_path: Path) -> None:
    long_action = " ".join(f"action{i}" for i in range(900))
    myservice_text = "\n".join(
        [
            "[meta]",
            json.dumps(
                {
                    "Title": "Pump maintenance",
                    "Model Name": "SUPRA XP",
                    "Equip_ID": "EQ-01",
                    "Order No.": "ORD-1",
                    "Activity Type": "Repair",
                    "Country": "KR",
                    "Reception Date": "2026-03-05",
                    "completeness": "full",
                },
                ensure_ascii=False,
                indent=2,
            ),
            "[status]",
            "현재 상태 요약",
            "[action]",
            long_action,
            "[cause]",
            "원인 설명",
            "[result]",
            "처리 결과",
        ]
    )
    txt_path = tmp_path / "myservice_case.txt"
    txt_path.write_text(myservice_text, encoding="utf-8")

    chunks = chunk_myservice(txt_path)

    assert len(chunks) >= 4
    assert all(c.doc_type == "myservice" for c in chunks)
    action_chunks = [c for c in chunks if c.extra_meta.get("section") == "action"]
    assert len(action_chunks) >= 2
    sections_present = chunks[0].extra_meta.get("sections_present", {})
    assert sections_present.get("status") is True
    assert sections_present.get("action") is True
    assert sections_present.get("cause") is True
    assert sections_present.get("result") is True


def test_chunk_gcb_tiered_summary_and_detail(tmp_path: Path) -> None:
    payload = [
        {
            "GCB_number": "123456",
            "Status": "Open",
            "Title": "Pump fault",
            "Model Name": "SUPRA XP",
            "Equip_ID": "EQ-99",
            "Request_Item2": "Inspection",
            "Content": "Description: issue overview\nCause: valve failure\nResult: replaced valve",
        }
    ]
    json_path = tmp_path / "gcb.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    chunks = chunk_gcb(json_path)

    assert chunks
    tiers = [c.extra_meta.get("chunk_tier") for c in chunks]
    assert tiers.count("summary") == 1
    assert tiers.count("detail") >= 1
    summary = next(c for c in chunks if c.extra_meta.get("chunk_tier") == "summary")
    assert summary.chapter == "summary"


def test_run_embedding_embed_model_uses_batch_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEmbedder:
        def encode(self, texts: list[str]) -> np.ndarray:
            vecs = np.zeros((len(texts), 1024), dtype=np.float32)
            vecs[:, 0] = 1.0
            return vecs

        def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
            vecs = np.zeros((len(texts), 1024), dtype=np.float32)
            vecs[:, 0] = 1.0
            return vecs

        def embed(self, texts: list[str]) -> np.ndarray:
            raise AssertionError("embed() should not be used")

    monkeypatch.setattr(run_embedding_module, "_create_embedder", lambda *_a, **_k: FakeEmbedder())

    vectors = embed_model("bge_m3", ["q1", "q2", "q3"], batch_size=2, device="cpu")
    assert vectors.shape == (3, 1024)


def test_chunk_v3_content_mapping_guardrails() -> None:
    mapping = get_chunk_v3_content_mapping()
    assert mapping.get("dynamic") is False
    extra_meta = mapping.get("properties", {}).get("extra_meta", {})
    assert extra_meta.get("type") == "object"
    assert extra_meta.get("enabled") is False
