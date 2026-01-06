import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from services.graph_ingest.enrichers import (
    canonical_key,
    extract_maintenance_bridges,
    extract_ts_bridges,
)
from services.graph_ingest.parsers.maintenance import MaintenanceDoc
from services.graph_ingest.parsers.ts import TSActionItem, TSDoc, TSStep


def _bridge_keys(bridges, label):
    return {bridge.key: bridge for bridge in bridges if bridge.label == label}


def _edge_signatures(edges, label):
    return {
        (edge.rel, edge.dst_label, edge.dst_id): edge
        for edge in edges
        if edge.dst_label == label
    }


def test_extract_ts_bridges_builds_steps_components_and_refs() -> None:
    doc = TSDoc(
        ts_id="TS-123",
        type="TSGuide",
        title="Trouble Shooting Guide - Trace Vacuum",
        categories={"A": "Fast vacuum timeout"},
        steps=[
            TSStep(
                code="A-1",
                title="CDA flow pressure",
                category="A",
                items=[
                    TSActionItem(
                        kind="check",
                        text="Check CDA flow pressure (spec. : 0.55~0.65MPa)",
                    )
                ],
            )
        ],
        meta={"references_by_issue": json.dumps({"A": ["• SOP-001 - CDA supply check"]})},
    )

    bridges, edges = extract_ts_bridges(doc)

    issue_keys = _bridge_keys(bridges, "Issue")
    assert "fast_vacuum_timeout" in issue_keys

    block_keys = _bridge_keys(bridges, "StepBlock")
    block_key = canonical_key("TS-123:A:steps")
    assert block_key in block_keys
    block_props = block_keys[block_key].props
    assert block_props["issue_code"] == "A"
    assert block_props["step_count"] == 1
    assert "CDA flow pressure" in block_props["text"]

    doc_ref_keys = _bridge_keys(bridges, "DocRef")
    assert "sop_001_cda_supply_check" in doc_ref_keys
    assert doc_ref_keys["sop_001_cda_supply_check"].props.get("kind") == "SOP"

    def _edge(rel: str, dst_label: str, dst_id: str, src_label: str | None = None) -> bool:
        return any(
            edge.rel == rel
            and edge.dst_label == dst_label
            and edge.dst_id == dst_id
            and (src_label is None or edge.src_label == src_label)
            for edge in edges
        )
    assert _edge("TROUBLE", "Issue", "fast_vacuum_timeout", src_label="TSGuide")
    assert _edge("HAS_STEP", "StepBlock", block_key, src_label="Issue")
    assert _edge("REFERS_TO", "DocRef", "sop_001_cda_supply_check", src_label="Issue")


def test_extract_maintenance_bridges_links_equipment_and_parts() -> None:
    doc = MaintenanceDoc(
        doc_id="20240101-001",
        type="MaintenanceLog",
        text="[status]\n-. 현상: FFU 압력 이상\n",
        meta={
            "Title": "FFU Fan Abnormal",
            "Equip No.": "PVK34708",
            "Model": "PRECIA",
            "Line": "SEC-P3(D)",
            "Parts": [{"No": "C0800975", "Name": "Manometer"}],
        },
        sections={"status": "-. 현상: FFU 압력 이상", "action": "", "cause": "", "result": ""},
    )

    bridges, edges = extract_maintenance_bridges(doc)

    bridge_map = _bridge_keys(bridges, "Equipment")
    assert "pvk34708" in bridge_map

    issue_map = _bridge_keys(bridges, "Issue")
    assert "ffu_fan_abnormal" in issue_map or "ffu" in issue_map

    component_map = _bridge_keys(bridges, "Component")
    assert "c0800975" in component_map

    equipment_edges = _edge_signatures(edges, "Equipment")
    assert ("AFFECTS_EQUIPMENT", "Equipment", "pvk34708") in equipment_edges

    component_edges = _edge_signatures(edges, "Component")
    assert ("USES_PART", "Component", "c0800975") in component_edges

    line_edges = _edge_signatures(edges, "ProductionLine")
    assert ("LOCATED_AT", "ProductionLine", "sec_p3_d") in line_edges
