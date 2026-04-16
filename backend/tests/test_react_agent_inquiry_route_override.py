import pytest
from backend.llm_infrastructure.llm.react_agent import _should_override_setup_to_general


class TestShouldOverrideSetupToGeneral:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("GENEVA 모델번호가 뭐야?", True),
            ("부품번호 알려줘", True),
            ("What is the part number for GENEVA?", True),
            ("형번을 알고 싶습니다", True),
            ("전체 목록을 보여줘", True),
            ("what are the supported models?", True),
            ("GENEVA 스펀在哪里?", True),
            ("어디서 찾을 수 있어?", True),
            ("where can I find the catalog?", True),
            ("이 모델 지원 가능해?", True),
            ("対応している型号は?", True),
            ("Is this tool supported?", True),
            ("도구가 뭐가 필요해?", True),
            ("What equipment do I need?", True),
            ("어떤 장비 필요해?", True),
            ("어떤 도구로 할 수 있어?", True),
            ("GENEVA 설치 방법 알려줘", False),
            ("설비 연결하는 법", False),
            ("如何使用 GENEVA?", False),
            ("how to setup the equipment", False),
            ("설정 방법", False),
            ("설치하는 법을 알려줘", False),
            ("조립/manual이 필요해", False),
            ("GENEVA에 대해 알려줘", False),
            ("도움이 필요해", False),
            ("help me", False),
            ("GENEVA 알람 원인", False),
            ("에러 점검 방법", False),
            ("이상 증상은?", False),
            ("troubleshoot Alba", False),
            ("alarm 해결 방법", False),
            ("문제 생겼어", False),
        ],
    )
    def test_should_override_setup_to_general(self, query: str, expected: bool) -> None:
        result = _should_override_setup_to_general(query)
        assert result == expected, f"Query: '{query}' - expected {expected}, got {result}"

    def test_procedure_keyword_takes_precedence(self) -> None:
        assert _should_override_setup_to_general("어떤 방법으로 설치해?") is False

    def test_english_procedure_keywords(self) -> None:
        assert _should_override_setup_to_general("How to install GENEVA?") is False
        assert _should_override_setup_to_general("setup guide for GENEVA") is False
        assert _should_override_setup_to_general("connect the equipment") is False
