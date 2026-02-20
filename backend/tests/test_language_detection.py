from backend.llm_infrastructure.llm.langgraph_agent import _detect_language_rule_based


def test_detect_language_rule_based_returns_korean() -> None:
    assert _detect_language_rule_based("SUPRA N 장비 캘리브레이션 방법 알려줘") == "ko"


def test_detect_language_rule_based_returns_english() -> None:
    assert _detect_language_rule_based("How to calibrate SUPRA N?") == "en"


def test_detect_language_rule_based_returns_japanese() -> None:
    assert _detect_language_rule_based("SUPRA Nのセットアップ手順を教えてください") == "ja"


def test_detect_language_rule_based_returns_chinese() -> None:
    assert _detect_language_rule_based("SUPRA N设备校准步骤是什么？") == "zh"
