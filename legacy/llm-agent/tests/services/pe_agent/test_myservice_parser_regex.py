from services.pe_agent.pe_core.parsers.myservice_parser import parse_maintenance_data


def test_parse_maintenance_data_handles_regex_bank_sections() -> None:
    ticket = {
        "Order No.": "20240001-001",
        "Description": (
            "1) 현상\n"
            "- 장비 온도 상승\n"
            "2) 조치\n"
            "- SOP-123에 따라 냉각 라인 점검\n"
            "3) 원인\n"
            "- 냉각수 부족\n"
            "4) 결과\n"
            "- 정상 복구"
        ),
    }

    parsed = parse_maintenance_data(ticket)

    assert parsed["status"].startswith("- 장비 온도 상승")
    assert parsed["action"].startswith("- SOP-123에 따라")
    assert parsed["cause"].startswith("- 냉각수 부족")
    assert parsed["result"].startswith("- 정상 복구")
    assert parsed["meta"]["completeness"] == "complete"
