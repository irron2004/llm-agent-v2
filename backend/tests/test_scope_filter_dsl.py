import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from collections.abc import Mapping
from typing import Callable, cast

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

SCOPE_FILTER_PATH = (
    BACKEND_ROOT / "llm_infrastructure" / "retrieval" / "filters" / "scope_filter.py"
)
spec = spec_from_file_location("scope_filter_module", SCOPE_FILTER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load scope filter module: {SCOPE_FILTER_PATH}")
scope_filter_module = module_from_spec(spec)
spec.loader.exec_module(scope_filter_module)

apply_scope_filter = cast(
    Callable[
        [Mapping[str, object] | None, Mapping[str, object] | None],
        dict[str, object] | None,
    ],
    scope_filter_module.apply_scope_filter,
)
build_scope_filter_by_doc_ids = cast(
    Callable[..., dict[str, object] | None],
    scope_filter_module.build_scope_filter_by_doc_ids,
)
build_scope_filter_by_fields = cast(
    Callable[..., dict[str, object]],
    scope_filter_module.build_scope_filter_by_fields,
)


def test_build_scope_filter_by_doc_ids_devices_only() -> None:
    actual = build_scope_filter_by_doc_ids(
        ["DEV-B", "DEV-A"],
        None,
        shared_doc_ids=["doc-2", "doc-1"],
        device_doc_types=["ts", "sop"],
        equip_doc_types=["myservice", "gcb"],
    )

    expected = {
        "bool": {
            "should": [
                {"terms": {"doc_id": ["doc-1", "doc-2"]}},
                {
                    "bool": {
                        "must": [
                            {"terms": {"doc_type": ["sop", "ts"]}},
                            {"terms": {"device_name": ["DEV-A", "DEV-B"]}},
                        ],
                        "must_not": [{"terms": {"doc_id": ["doc-1", "doc-2"]}}],
                    }
                },
                {
                    "bool": {
                        "must": [
                            {"terms": {"doc_type": ["gcb", "myservice"]}},
                            {"terms": {"device_name": ["DEV-A", "DEV-B"]}},
                        ]
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }
    assert actual == expected


def test_build_scope_filter_by_doc_ids_equip_only() -> None:
    actual = build_scope_filter_by_doc_ids(
        [],
        ["eq-2", "eq-1"],
        shared_doc_ids=["doc-2", "doc-1"],
        device_doc_types=["ts", "sop"],
        equip_doc_types=["myservice", "gcb"],
    )

    expected = {
        "bool": {
            "should": [
                {"terms": {"doc_id": ["doc-1", "doc-2"]}},
                {
                    "bool": {
                        "must": [
                            {"terms": {"doc_type": ["gcb", "myservice"]}},
                            {"terms": {"equip_id": ["eq-1", "eq-2"]}},
                        ]
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }
    assert actual == expected


def test_build_scope_filter_by_doc_ids_devices_and_equip() -> None:
    actual = build_scope_filter_by_doc_ids(
        ["DEV-B", "DEV-A"],
        ["eq-2", "eq-1"],
        shared_doc_ids=["doc-2", "doc-1"],
        device_doc_types=["ts", "sop"],
        equip_doc_types=["myservice", "gcb"],
    )

    expected = {
        "bool": {
            "should": [
                {"terms": {"doc_id": ["doc-1", "doc-2"]}},
                {
                    "bool": {
                        "must": [
                            {"terms": {"doc_type": ["sop", "ts"]}},
                            {"terms": {"device_name": ["DEV-A", "DEV-B"]}},
                        ],
                        "must_not": [{"terms": {"doc_id": ["doc-1", "doc-2"]}}],
                    }
                },
                {
                    "bool": {
                        "must": [
                            {"terms": {"doc_type": ["gcb", "myservice"]}},
                            {"terms": {"equip_id": ["eq-1", "eq-2"]}},
                        ]
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }
    assert actual == expected


def test_build_scope_filter_by_doc_ids_empty_scope_returns_none() -> None:
    actual = build_scope_filter_by_doc_ids(
        [],
        [],
        shared_doc_ids=[],
        device_doc_types=["sop", "ts"],
        equip_doc_types=["myservice", "gcb"],
    )
    assert actual is None


def test_build_scope_filter_by_fields_structure() -> None:
    actual = build_scope_filter_by_fields(["DEV-B", "DEV-A"], ["eq-2", "eq-1"])

    expected = {
        "bool": {
            "should": [
                {
                    "bool": {
                        "must": [
                            {"term": {"scope_level": "shared"}},
                            {"term": {"is_shared": True}},
                        ]
                    }
                },
                {
                    "bool": {
                        "must": [
                            {"term": {"scope_level": "device"}},
                            {"terms": {"device_name": ["DEV-A", "DEV-B"]}},
                        ]
                    }
                },
                {
                    "bool": {
                        "must": [
                            {"term": {"scope_level": "equip"}},
                            {"terms": {"equip_id": ["eq-1", "eq-2"]}},
                        ]
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }
    assert actual == expected


def test_apply_scope_filter_combines_with_bool_filter() -> None:
    base_filter = {"term": {"tenant_id": "tenant-a"}}
    scope_filter = {"terms": {"doc_id": ["doc-1"]}}

    actual = apply_scope_filter(base_filter, scope_filter)
    expected = {"bool": {"filter": [base_filter, scope_filter]}}

    assert actual == expected
