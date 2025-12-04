from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import services.graph_ingest.parsers.maintenance as maintenance


@dataclass
class _SampleDataclass:
    value: int


class _CustomObject:
    def __init__(self) -> None:
        self.payload = {"inner": {"nested": 1}}


def _stub_parser(_: dict) -> dict:
    return {
        "meta": {
            "Order No.": "MNT-1",
            "Title": "Example",
            "numeric": 42,
            "set_field": {3, 1, 2},
            "tuple_field": ("a", _CustomObject()),
            "dict_field": {"with_set": {"values": {"x", "y"}}},
            "Parts": [{"No": "123", "Name": "Filter"}],
            7: _SampleDataclass(5),
            None: "drop-me",
        },
        "status": "ok",
        "action": "act",
        "cause": None,
        "result": "done",
    }


def test_parse_maintenance_json_item_sanitizes_meta(monkeypatch):
    monkeypatch.setattr(maintenance, "parse_maintenance_data", _stub_parser)

    doc = maintenance.parse_maintenance_json_item({})

    assert doc.doc_id == "MNT-1"
    assert doc.meta["Title"] == "Example"
    assert doc.meta["numeric"] == 42
    assert doc.meta["set_field"] == [1, 2, 3]
    assert doc.meta["tuple_field"][0] == "a"
    assert doc.meta["tuple_field"][1] == {"payload": {"inner": {"nested": 1}}}
    assert doc.meta["dict_field"] == {"with_set": {"values": ["x", "y"]}}
    assert doc.meta["Parts"] == [{"No": "123", "Name": "Filter"}]
    assert doc.meta["7"] == {"value": 5}
    assert None not in doc.meta


def test_parse_maintenance_json_sanitizes_each_item(monkeypatch):
    call_count = {"value": 0}

    def parser(payload: dict) -> dict:
        call_count["value"] += 1
        return {
            "meta": {"Order No.": f"{payload['id']}"},
            "status": "s",
            "action": "a",
            "cause": "c",
            "result": "r",
        }

    monkeypatch.setattr(maintenance, "parse_maintenance_data", parser)

    docs = maintenance.parse_maintenance_json([
        {"id": 1},
        {"id": 2},
    ])

    assert [doc.doc_id for doc in docs] == ["1", "2"]
    assert call_count["value"] == 2
