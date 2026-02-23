from backend.api.routers.agent import AutoParseResult, _should_suggest_additional_device_search


def test_should_suggest_false_when_auto_parse_not_present() -> None:
    result = {}

    assert _should_suggest_additional_device_search(result, None) is False


def test_should_suggest_true_when_auto_parse_has_no_device() -> None:
    result = {
        "auto_parsed_doc_type": "gcb",
        "auto_parse_message": "파싱 결과 - 문서: gcb, lang: kor",
    }
    auto_parse = AutoParseResult(
        doc_type="gcb",
        message="파싱 결과 - 문서: gcb, lang: kor",
    )

    assert _should_suggest_additional_device_search(result, auto_parse) is True


def test_should_suggest_false_when_device_detected() -> None:
    result = {
        "auto_parsed_device": "SUPRA N",
        "auto_parsed_devices": ["SUPRA N"],
    }
    auto_parse = AutoParseResult(
        device="SUPRA N",
        devices=["SUPRA N"],
    )

    assert _should_suggest_additional_device_search(result, auto_parse) is False


def test_should_suggest_true_when_only_equip_id_detected() -> None:
    result = {
        "auto_parsed_equip_id": "EPAG50",
        "auto_parsed_equip_ids": ["EPAG50"],
    }
    auto_parse = AutoParseResult(
        equip_id="EPAG50",
        equip_ids=["EPAG50"],
    )

    assert _should_suggest_additional_device_search(result, auto_parse) is True


def test_should_suggest_true_when_selected_devices_exists_but_auto_parse_device_missing() -> None:
    result = {
        "selected_devices": ["SUPRA N"],
        "auto_parse_message": "파싱 결과 - lang: kor",
        "detected_language": "ko",
    }
    auto_parse = AutoParseResult(
        language="ko",
        message="파싱 결과 - lang: kor",
    )

    assert _should_suggest_additional_device_search(result, auto_parse) is True


def test_should_suggest_true_when_auto_parse_markers_exist_without_auto_parse_object() -> None:
    result = {
        "auto_parsed_device": None,
        "auto_parsed_devices": [],
        "detected_language": "ko",
        "auto_parse_message": "파싱 결과 - lang: kor",
    }

    assert _should_suggest_additional_device_search(result, None) is True


def test_should_suggest_true_when_only_apc_is_parsed_as_device() -> None:
    result = {
        "auto_parsed_device": "APC",
        "auto_parsed_devices": ["APC"],
        "auto_parse_message": "Parsed - Device: APC, lang: eng",
        "detected_language": "en",
    }
    auto_parse = AutoParseResult(
        device="APC",
        devices=["APC"],
        language="en",
        message="Parsed - Device: APC, lang: eng",
    )

    assert _should_suggest_additional_device_search(result, auto_parse) is True
