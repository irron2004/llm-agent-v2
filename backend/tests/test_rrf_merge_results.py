import sys
import importlib.util
import types
from pathlib import Path
from typing import Protocol, cast

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
RETRIEVAL_ROOT = BACKEND_ROOT / "llm_infrastructure" / "retrieval"

PACKAGE_NAME = "_rrf_testpkg"
_pkg = types.ModuleType(PACKAGE_NAME)
_pkg.__path__ = [str(RETRIEVAL_ROOT)]
sys.modules[PACKAGE_NAME] = _pkg

_base_spec = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.base",
    RETRIEVAL_ROOT / "base.py",
)
assert _base_spec is not None and _base_spec.loader is not None
_base_module = importlib.util.module_from_spec(_base_spec)
sys.modules[_base_spec.name] = _base_module
_base_spec.loader.exec_module(_base_module)
RetrievalResult = cast(type, getattr(_base_module, "RetrievalResult"))


class _ResultLike(Protocol):
    doc_id: str
    metadata: dict[str, object] | None
    score: float


class _MergeFn(Protocol):
    def __call__(
        self,
        stage1: list[object],
        stage2: list[object],
        *,
        k: int = 60,
    ) -> list[_ResultLike]: ...


RRF_MODULE_PATH = RETRIEVAL_ROOT / "rrf.py"
_rrf_spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.rrf", RRF_MODULE_PATH)
assert _rrf_spec is not None and _rrf_spec.loader is not None
_rrf_module = importlib.util.module_from_spec(_rrf_spec)
sys.modules[_rrf_spec.name] = _rrf_module
_rrf_spec.loader.exec_module(_rrf_module)
merge_retrieval_results_rrf = cast(_MergeFn, getattr(_rrf_module, "merge_retrieval_results_rrf"))


def test_rrf_merge_is_deterministic_across_repeated_runs() -> None:
    stage1 = [
        RetrievalResult(
            doc_id="doc-z",
            content="z1",
            score=1.0,
            metadata={"page": "10"},
        ),
        RetrievalResult(
            doc_id="doc-a",
            content="a1",
            score=0.9,
            metadata={"chunk_id": "2", "page": "3"},
        ),
    ]
    stage2 = [
        RetrievalResult(
            doc_id="doc-a",
            content="a2",
            score=1.0,
            metadata={"chunk_id": "2", "page": "5"},
        ),
        RetrievalResult(
            doc_id="doc-z",
            content="z2",
            score=0.9,
            metadata={"page": "10"},
        ),
    ]

    orders: list[list[tuple[str, str, str]]] = []
    for _ in range(5):
        merged = merge_retrieval_results_rrf(stage1, stage2)
        orders.append(
            [
                (
                    item.doc_id,
                    str((item.metadata or {}).get("page")),
                    str((item.metadata or {}).get("chunk_id")),
                )
                for item in merged
            ]
        )

    assert all(order == orders[0] for order in orders)


def test_rrf_merge_tie_break_uses_doc_page_chunk_order() -> None:
    merged_doc_id_tie = merge_retrieval_results_rrf(
        stage1=[RetrievalResult(doc_id="doc-b", content="b", score=1.0)],
        stage2=[RetrievalResult(doc_id="doc-a", content="a", score=1.0)],
    )
    assert [item.doc_id for item in merged_doc_id_tie] == ["doc-a", "doc-b"]

    merged_page_tie = merge_retrieval_results_rrf(
        stage1=[
            RetrievalResult(
                doc_id="doc-p",
                content="p10",
                score=1.0,
                metadata={"page": "10"},
            )
        ],
        stage2=[
            RetrievalResult(
                doc_id="doc-p",
                content="p2",
                score=1.0,
                metadata={"page": "2"},
            )
        ],
    )
    assert [(item.metadata or {}).get("page") for item in merged_page_tie] == ["2", "10"]

    merged_chunk_tie = merge_retrieval_results_rrf(
        stage1=[
            RetrievalResult(
                doc_id="doc-c",
                content="c10",
                score=1.0,
                metadata={"page": 1, "chunk_id": "10"},
            )
        ],
        stage2=[
            RetrievalResult(
                doc_id="doc-c",
                content="c2",
                score=1.0,
                metadata={"page": 1, "chunk_id": "2"},
            )
        ],
    )
    assert [(item.metadata or {}).get("chunk_id") for item in merged_chunk_tie] == ["2", "10"]


def test_rrf_merge_dedupes_same_doc_and_chunk_id() -> None:
    stage1 = [
        RetrievalResult(
            doc_id="doc-1",
            content="stage1",
            score=0.8,
            metadata={"chunk_id": "chunk-7", "page": 4},
        )
    ]
    stage2 = [
        RetrievalResult(
            doc_id="doc-1",
            content="stage2-best",
            score=0.95,
            metadata={"chunk_id": "chunk-7", "page": 9},
        )
    ]

    merged = merge_retrieval_results_rrf(stage1, stage2)

    assert len(merged) == 1
    assert merged[0].doc_id == "doc-1"
    assert merged[0].metadata == {"chunk_id": "chunk-7", "page": 4}
    assert merged[0].score == (1.0 / 61.0) + (1.0 / 61.0)
