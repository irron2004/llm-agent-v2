import pytest

# 테스트 실행 시 repo 루트를 PYTHONPATH에 추가
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.preprocessing.normalize_engine import (
    NormLevel,
    build_normalizer,
    clean_variants,
    normalize_text,
    preprocess_l4_advanced_domain,
    preprocess_l5_enhanced_domain,
    preprocess_semiconductor_domain,
    prejoin_domain_terms,
    sanitize_variant_map,
)
from backend.llm_infrastructure.preprocessing.adapters.normalize import (
    NormalizationPreprocessor,
)
from backend.llm_infrastructure.preprocessing.adapters.standard import (
    StandardPreprocessor,
)
from backend.llm_infrastructure.preprocessing.adapters.domain_specific import (
    DomainSpecificPreprocessor,
)
from backend.llm_infrastructure.preprocessing.registry import (
    PreprocessorRegistry,
    get_preprocessor,
)


# --- base utilities ---


def test_normalize_text_basic_and_newlines():
    text = "A\n\nB   C µm"
    keep = normalize_text(text, keep_newlines=True)
    drop = normalize_text(text, keep_newlines=False)
    assert keep == "a\nb c um"
    assert drop == "a b c um"


def test_clean_variants_respects_boundaries_and_case():
    variant_map = {"pm": "PMX", "ABC": "xyz"}
    src = "pm2-1 pm abc"
    out = clean_variants(src, variant_map)
    # 내부 숫자/문자에 붙은 pm은 그대로, 단독 pm/abc는 치환
    assert out == "pm2-1 PMX xyz"


def test_sanitize_variant_map_filters_ambiguous_and_short():
    raw = {
        "test": "skip",  # 모호어
        "pc": "particle",  # 짧은 용어
        "abc": "ABC",  # 길이 3 (필터)
        "AB12": "keep",  # 약어 패턴 허용
    }
    safe = sanitize_variant_map(raw)
    assert "ab12" in safe and safe["ab12"] == "keep"
    assert "abc" not in safe
    assert "test" not in safe
    assert "pc" not in safe


# --- domain helpers ---


def test_prejoin_domain_terms_combines_known_patterns():
    text = "ball screw at pm 2-1 with ll 1"
    joined = prejoin_domain_terms(text)
    assert "ball_screw" in joined
    assert "pm2_1" in joined
    assert "ll1" in joined


def test_preprocess_semiconductor_domain_masks_pm_and_scientific_and_variants():
    text = "pm2-1 helium leak 4.0x10^3 exhasut"
    out = preprocess_semiconductor_domain(text, {"exhaust": "EX"})
    # pm 주소 마스킹, 과학 표기 변환, 오탈자 수정, 변형어 치환
    assert out == "PM helium leak 4000.0 EX"


def test_preprocess_l4_advanced_domain_adds_header_tokens():
    text = "pm 2 alarm (1234) spec out helium leak 4.0x10^-9 mt 5"
    out = preprocess_l4_advanced_domain(text)
    header = "[MODULE PM2] [ALARM 1234] [SPEC OUT] [HE_LEAK 4.00e-09] [PRESS_MT_MAX 5]"
    assert out.startswith(header)
    assert "::" in out
    # 본문은 그대로 포함
    assert "spec out helium leak 4.0x10^-9 mt 5" in out


def test_preprocess_l5_enhanced_domain_applies_variants_range_and_tokens():
    text = "pm 1 spec out 15 mt backup 10~20"
    out = preprocess_l5_enhanced_domain(text)
    header = "[MODULE PM1] [SPEC OUT] [PRESS_MT_MAX 15] [ACTION BKUP]"
    assert out.startswith(header)
    assert "[RANGE 10.0..20.0]" in out
    # L5 변형어 맵이 적용되어 spec out/backup이 대문자화되었는지 확인
    assert "SPEC OUT" in out
    assert "BACKUP" in out


# --- factory ---


def test_build_normalizer_level_selection_and_profile():
    vm = {"pm": "PMX"}
    n0 = build_normalizer("L0")
    n1 = build_normalizer("L1", vm, keep_newlines=False)
    n3 = build_normalizer("L3")

    assert n0("PM 2") == "pm 2"
    assert n1("pm here") == "PMX here"
    assert n3("pm 2-1 alarm")  # returns non-empty processed string

    profile = getattr(n1, "__safe_profile__", {})
    assert profile.get("level") == "L1"
    assert profile.get("sanitized_variants") == len(vm)
    assert profile.get("keep_newlines") is False
    assert profile.get("semiconductor_domain") is False


# --- adapters ---


def test_normalization_preprocessor_preserves_metadata():
    pre = NormalizationPreprocessor(level="L0", keep_newlines=False)
    docs = [{"content": " PM 2 ", "id": 1}, " pm "]
    results = list(pre.preprocess(docs))
    assert results[0]["content"] == "pm 2"
    assert results[0]["id"] == 1  # 메타데이터 유지
    assert results[1] == "pm"


def test_domain_specific_preprocessor_units_and_abbreviations():
    pre = DomainSpecificPreprocessor(
        normalize_units=True,
        expand_abbreviations=True,
    )
    docs = ["Temp 100 °C pres 50 psi"]
    out = list(pre.preprocess(docs))
    assert out == ["temperature 100 degC pressure 50 PSI"]


# --- registry ---


def test_registry_returns_registered_preprocessors():
    # 데코레이터로 등록된 기본 전처리기가 조회되는지 확인
    normalize_proc = get_preprocessor("normalize", version="v1")
    standard_proc = get_preprocessor("standard", version="v1")
    domain_proc = get_preprocessor("pe_domain", version="v1")

    assert isinstance(normalize_proc, NormalizationPreprocessor)
    assert isinstance(standard_proc, StandardPreprocessor)
    assert isinstance(domain_proc, DomainSpecificPreprocessor)

    methods = PreprocessorRegistry.list_methods()
    for name in ["normalize", "standard", "pe_domain"]:
        assert name in methods
