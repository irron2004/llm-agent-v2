import pytest

from services.graph_ingest.linker import extract_issue_hints, extract_param_ranges, extract_refs
from services.graph_ingest.parsers.maintenance import MaintenanceDoc, parse_maintenance_json_item
from services.graph_ingest.parsers.qa import parse_qa_rows
from services.graph_ingest.validation import validate_maintenance


def test_parse_maintenance_json_item_builds_doc() -> None:
    item = {
        "meta": {
            "Order No.": "20240101-001",
            "completeness": "complete",
            "sections_present": {"status": True, "action": True, "cause": True, "result": True},
        },
        "status": "완료",
        "action": "조치",
        "cause": "원인",
        "result": "결과",
    }

    doc = parse_maintenance_json_item(item)

    assert doc.doc_id == "20240101-001"
    assert doc.type == "MaintenanceLog"
    assert "[status]" in doc.text
    assert doc.sections["status"] == "완료"


def test_parse_maintenance_json_item_requires_id() -> None:
    item = {
        "meta": {
            "completeness": "complete",
            "sections_present": {"status": True, "action": True, "cause": True, "result": True},
        },
        "status": "완료",
        "action": "조치",
        "cause": "원인",
        "result": "결과",
    }

    with pytest.raises(ValueError):
        parse_maintenance_json_item(item)


def test_extract_refs_finds_sop_and_ts_edges() -> None:
    text = """정비 진행 중 SOP-1234 를 참조 하십시오. 추가 점검은 TS 4321 refer to section 4."""

    edges = extract_refs(text, src_id="20240101-001", src_label="MaintenanceLog")

    assert len(edges) == 2
    sop_edge = next(edge for edge in edges if edge.dst_label == "SOP")
    assert sop_edge.dst_id == "SOP-1234"
    assert sop_edge.rel == "USES_PROCEDURE"
    assert sop_edge.props["source"] == "regex-bank-v1"


def test_extract_issue_hints_uses_seed_keywords() -> None:
    text = "엔진 과열 문제로 장비가 중단되었습니다."

    issues = extract_issue_hints(text)

    assert [issue.key for issue in issues] == ["엔진 과열"]


def test_extract_param_ranges_applies_whitelist() -> None:
    texts = [
        "압력: 3.5, 습도: 10",
        "온도 = -2",
        "A값 7",
        "압력 3.1-4.2 um",
        "온도 85/76",
    ]

    ranges = extract_param_ranges(texts)

    assert ranges["압력"] == (3.1, 4.2)
    assert ranges["온도"] == (-2.0, 85.0)
    assert "습도" not in ranges


def test_parse_qa_rows_assigns_defaults() -> None:
    rows = [
        {"question": "무엇?", "answer": "이것."},
        {"qa_id": "QA-42", "question": "왜?", "answer": "그러므로."},
    ]

    docs = parse_qa_rows(rows)

    assert docs[0].qa_id == "QA-00001"
    assert "[Q]" in docs[0].text
    assert docs[1].qa_id == "QA-42"


def test_validate_maintenance_flags_missing_status() -> None:
    doc = MaintenanceDoc(
        doc_id="20240101-001",
        type="MaintenanceLog",
        text="",
        meta={"completeness": "partial"},
        sections={"status": "", "action": "", "cause": "", "result": ""},
    )

    result = validate_maintenance(doc)

    assert "Some sections are missing" in result.warnings
    assert "Missing status text" in result.warnings


def test_extract_refs_cites_from_error_qa() -> None:
    text = "as per SOP-777 follow steps"

    edges = extract_refs(text, src_id="QA-123", src_label="ErrorQA")

    assert edges
    edge = edges[0]
    assert edge.dst_id == "SOP-777"
    assert edge.rel == "CITES"


def test_extract_refs_cross_ref_sentence() -> None:
    text = "please 참조 SOP No: HPS-320-R0 for procedures."

    edges = extract_refs(text, src_id="20240101-001", src_label="MaintenanceLog")

    assert edges
    edge = edges[0]
    assert edge.dst_label == "SOP"
    assert edge.dst_id == "HPS-320-R0"
    assert edge.props.get("pattern") in {"REF_SOP_FWD", "cross_ref_sentence", "target_label"}


def test_extract_param_ranges_handles_commas_and_ineq() -> None:
    texts = [
        "압력: 3.5",
        "Pressure > 500,000 mT",
        "온도는 23.76V~24.72V",
    ]

    ranges = extract_param_ranges(texts)

    assert ranges["압력"] == (3.5, 3.5)
    assert ranges["온도"][1] == 24.72
