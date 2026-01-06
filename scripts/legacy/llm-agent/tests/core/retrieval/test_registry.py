"""
Test Parser Registry

Parser Registry의 기본 기능을 테스트합니다.
"""

import pytest
from core.retrieval.registry import (
    register_parser, 
    get_parser, 
    list_parsers, 
    clear_parsers
)

def test_register_and_get_parser():
    """파서 등록 및 조회 테스트"""
    # 테스트용 더미 파서
    def dummy_parser(file_path: str):
        return [("test text", "test_id")]
    
    # 파서 등록
    register_parser("test_parser", dummy_parser)
    
    # 파서 조회
    retrieved_parser = get_parser("test_parser")
    
    # 동일한 함수인지 확인
    assert retrieved_parser is dummy_parser
    
    # 파서 실행 테스트
    result = retrieved_parser("dummy_path")
    assert result == [("test text", "test_id")]

def test_get_nonexistent_parser():
    """존재하지 않는 파서 조회 시 오류 테스트"""
    with pytest.raises(ValueError, match="Unknown parser"):
        get_parser("nonexistent_parser")

def test_list_parsers():
    """등록된 파서 목록 조회 테스트"""
    # 기존 파서들 정리
    clear_parsers()
    
    # 파서 등록
    def parser1(file_path: str):
        return [("text1", "id1")]
    
    def parser2(file_path: str):
        return [("text2", "id2")]
    
    register_parser("parser1", parser1)
    register_parser("parser2", parser2)
    
    # 파서 목록 조회
    parsers = list_parsers()
    
    # 정렬된 순서로 반환되는지 확인
    assert parsers == ["parser1", "parser2"]

def test_clear_parsers():
    """파서 목록 정리 테스트"""
    # 파서 등록
    def dummy_parser(file_path: str):
        return [("text", "id")]
    
    register_parser("dummy", dummy_parser)
    
    # 파서 목록 정리
    clear_parsers()
    
    # 파서 목록이 비어있는지 확인
    assert list_parsers() == []

def test_parser_validation():
    """파서 유효성 검사 테스트"""
    # callable이 아닌 객체 등록 시 오류
    with pytest.raises(ValueError, match="Parser must be callable"):
        register_parser("invalid", "not_callable")
    
    # None 등록 시 오류
    with pytest.raises(ValueError, match="Parser must be callable"):
        register_parser("invalid", None)

def test_parser_protocol_compliance():
    """파서 프로토콜 준수 테스트"""
    # 올바른 시그니처를 가진 파서
    def valid_parser(file_path: str):
        return [("text", "id")]
    
    # 잘못된 시그니처를 가진 파서 (경고만, 오류는 아님)
    def invalid_parser(wrong_param):
        return [("text", "id")]
    
    # 둘 다 등록 가능해야 함 (런타임에만 검증)
    register_parser("valid", valid_parser)
    register_parser("invalid", invalid_parser)
    
    # 등록된 파서들
    assert "valid" in list_parsers()
    assert "invalid" in list_parsers()

if __name__ == "__main__":
    pytest.main([__file__])
