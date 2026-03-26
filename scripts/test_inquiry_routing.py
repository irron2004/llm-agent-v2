"""비절차 질문 라우팅 테스트 스크립트.

SOP/Setup 문서 선택 시 비절차 질문이 올바르게 route=general로 분류되고,
절차 질문은 route=setup으로 유지되는지 검증한다.
"""
import json
import sys
import requests

API = "http://localhost:8001/api/agent/run"

TESTS = [
    # === Setup 문서 비절차 질문 ===
    {
        "name": "Setup: check sheet 조회",
        "message": "setup check sheet 조회해줘",
        "filter_doc_types": ["setup"],
        "filter_devices": ["SUPRA_N"],
        "expect_route": "general",
    },
    {
        "name": "Setup: teaching 포인트 목록",
        "message": "teaching 포인트 목록 보여줘",
        "filter_doc_types": ["setup"],
        "filter_devices": ["SUPRA_N"],
        "expect_route": "general",
    },
    {
        "name": "Setup: TTTM 매칭 항목",
        "message": "TTTM 매칭 항목 리스트 알려줘",
        "filter_doc_types": ["setup"],
        "filter_devices": ["SUPRA_N"],
        "expect_route": "general",
    },
    # === SOP 비절차 질문 ===
    {
        "name": "SOP: part 위치",
        "message": "part 위치 보여줘",
        "filter_doc_types": ["sop"],
        "filter_devices": ["INTEGER_PLUS"],
        "expect_route": "general",
    },
    {
        "name": "SOP: 필요 tool 목록",
        "message": "필요 tool 목록 알려줘",
        "filter_doc_types": ["sop"],
        "filter_devices": ["SUPRA VPLUS"],
        "expect_route": "general",
    },
    {
        "name": "SOP: 작업자 위치",
        "message": "작업자 위치 조회",
        "filter_doc_types": ["sop"],
        "filter_devices": ["INTEGER_PLUS"],
        "expect_route": "general",
    },
    {
        "name": "SOP: 작업 체크시트",
        "message": "작업 check sheet 조회해줘",
        "filter_doc_types": ["sop"],
        "filter_devices": ["SUPRA_N"],
        "expect_route": "general",
    },
    {
        "name": "SOP: 사고 사례",
        "message": "bubbler cabinet 사고 사례 알려줘",
        "filter_doc_types": ["sop"],
        "filter_devices": ["GENEVA_XP"],
        "expect_route": "general",
    },
    {
        "name": "SOP: 안전 보호구 체크시트",
        "message": "환경 안전 보호구 체크시트 조회해줘",
        "filter_doc_types": ["sop"],
        "filter_devices": ["SUPRA_N"],
        "expect_route": "general",
    },
    {
        "name": "SOP: flow chart",
        "message": "flow chart 보여줘",
        "filter_doc_types": ["sop"],
        "filter_devices": ["SUPRA VPLUS"],
        "expect_route": "general",
    },
    # === 절차 질문 (대조군 - setup 유지 확인) ===
    {
        "name": "절차 대조군: 교체 절차",
        "message": "slot valve 교체 절차 알려줘",
        "filter_doc_types": ["sop"],
        "filter_devices": ["SUPRA VPLUS"],
        "expect_route": "setup",
    },
    {
        "name": "절차 대조군: 설치 방법",
        "message": "sensor board 설치 방법",
        "filter_doc_types": ["sop"],
        "filter_devices": ["INTEGER_PLUS"],
        "expect_route": "setup",
    },
]


def run_test(test: dict, idx: int) -> dict:
    payload = {
        "message": test["message"],
        "filter_doc_types": test["filter_doc_types"],
        "filter_devices": test["filter_devices"],
        "top_k": 5,
        "mode": "base",
        "auto_parse": False,
    }
    try:
        resp = requests.post(API, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"name": test["name"], "status": "ERROR", "error": str(e)}

    meta = data.get("metadata", {})
    route = meta.get("route", "?")
    expect = test["expect_route"]
    ok = route == expect

    docs = data.get("retrieved_docs", [])
    chapters = []
    for d in docs[:3]:
        m = d.get("metadata", {})
        chapters.append(m.get("section_chapter", "?"))

    answer = (data.get("answer") or "")[:200]

    return {
        "name": test["name"],
        "status": "PASS" if ok else "FAIL",
        "route": route,
        "expect": expect,
        "chapters": chapters,
        "answer_preview": answer,
    }


if __name__ == "__main__":
    results = []
    for i, test in enumerate(TESTS):
        print(f"[{i+1}/{len(TESTS)}] {test['name']}...", end=" ", flush=True)
        r = run_test(test, i)
        print(r["status"])
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    print(f"PASS: {passed}  FAIL: {failed}  ERROR: {errors}  TOTAL: {len(results)}")
    print()

    for r in results:
        icon = "✓" if r["status"] == "PASS" else "✗" if r["status"] == "FAIL" else "!"
        print(f"  {icon} {r['name']}")
        if r["status"] == "ERROR":
            print(f"    error: {r.get('error', '')}")
        else:
            print(f"    route={r['route']} (expect={r['expect']})")
            print(f"    chapters: {r.get('chapters', [])}")
            print(f"    answer: {r.get('answer_preview', '')[:120]}...")
        print()
