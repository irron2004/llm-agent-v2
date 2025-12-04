"""
Test Parser Integration with Core Retrieval

파서 등록부터 코어 호출까지의 통합 테스트입니다.
"""

import pytest
import tempfile
import json
from pathlib import Path

def test_myservice_parser_registration():
    """MyService 파서가 레지스트리에 등록되는지 테스트"""
    # 파서 등록 모듈 임포트
    import pe_agent.evaluation.bootstrap_retrieval
    
    # 레지스트리에서 파서 확인
    from core.retrieval.registry import get_parser, list_parsers
    
    # myservice 파서가 등록되어 있는지 확인
    assert "myservice" in list_parsers()
    
    # 파서 함수 가져오기
    parser = get_parser("myservice")
    assert callable(parser)

def test_myservice_parser_functionality():
    """MyService 파서의 기능 테스트"""
    from core.retrieval.registry import get_parser
    
    # 파서 가져오기
    parser = get_parser("myservice")
    
    # 테스트용 JSON 데이터 생성
    test_data = {
        "item1": {
            "status": "완료",
            "action": "설치",
            "cause": "정기 점검",
            "result": "정상 작동",
            "meta": {"Order No.": "123456789-0001"}
        },
        "item2": {
            "status": "진행중",
            "action": "수리",
            "cause": "고장",
            "result": "수리 중",
            "meta": {"Order No.": "987654321-0002"}
        }
    }
    
    # 임시 파일에 테스트 데이터 저장
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, ensure_ascii=False)
        temp_file = f.name
    
    try:
        # 파서 실행
        result = parser(temp_file)
        
        # 결과 검증
        assert len(result) == 2
        
        # 첫 번째 문서 검증
        text1, doc_id1 = result[0]
        assert "[status]: 완료" in text1
        assert "[action]: 설치" in text1
        assert "[cause]: 정기 점검" in text1
        assert "[result]: 정상 작동" in text1
        assert doc_id1 == "123456789-0001"
        
        # 두 번째 문서 검증
        text2, doc_id2 = result[1]
        assert "[status]: 진행중" in text2
        assert "[action]: 수리" in text2
        assert "[cause]: 고장" in text2
        assert "[result]: 수리 중" in text2
        assert doc_id2 == "987654321-0002"
        
    finally:
        # 임시 파일 정리
        Path(temp_file).unlink(missing_ok=True)

def test_core_retrieval_import():
    """코어 retrieval 모듈들이 정상적으로 임포트되는지 테스트"""
    # 주요 모듈들 임포트 테스트
    from core.retrieval.api import Embedder, Retriever, Parser
    from core.retrieval.registry import register_parser, get_parser
    from core.retrieval.hybrid import HybridSearch
    from core.retrieval.evaluator import RetrievalEvaluator
    from core.retrieval.metrics import ReciprocalRankFusion, WeightedSum
    
    # 클래스들이 정의되어 있는지 확인
    assert Embedder is not None
    assert Retriever is not None
    assert Parser is not None
    assert HybridSearch is not None
    assert RetrievalEvaluator is not None
    assert ReciprocalRankFusion is not None
    assert WeightedSum is not None

def test_parser_protocol_compliance():
    """파서가 프로토콜을 준수하는지 테스트"""
    from core.retrieval.api import Parser
    from core.retrieval.registry import get_parser
    
    # myservice 파서 가져오기
    parser = get_parser("myservice")
    
    # 프로토콜 준수 확인
    assert hasattr(parser, '__call__')
    assert callable(parser)
    
    # 시그니처 확인 (런타임에만 가능)
    import inspect
    sig = inspect.signature(parser)
    params = list(sig.parameters.keys())
    
    # file_path 매개변수가 있는지 확인
    assert len(params) >= 1
    assert 'file_path' in params or params[0] == 'file_path'

def test_end_to_end_workflow():
    """전체 워크플로우 테스트 (간단한 버전)"""
    # 이 테스트는 실제 임베딩 모델 없이 기본 구조만 테스트
    
    # 1. 파서 등록 확인
    from core.retrieval.registry import list_parsers
    assert "myservice" in list_parsers()
    
    # 2. 기본 클래스들 생성 가능 확인
    from core.retrieval.metrics import ReciprocalRankFusion
    
    # RRF 메트릭 생성
    rrf = ReciprocalRankFusion(k=60.0)
    assert rrf is not None
    assert rrf.k == 60.0
    
    # 3. 기본 융합 로직 테스트
    dense_results = [
        {'doc_id': 'doc1', 'score': 0.9, 'text': 'text1'},
        {'doc_id': 'doc2', 'score': 0.8, 'text': 'text2'}
    ]
    
    sparse_results = [
        {'doc_id': 'doc2', 'score': 0.85, 'text': 'text2'},
        {'doc_id': 'doc1', 'score': 0.75, 'text': 'text1'}
    ]
    
    # 융합 실행
    fused = rrf.fuse(dense_results, sparse_results)
    
    # 결과 검증
    assert len(fused) == 2
    assert fused[0]['doc_id'] in ['doc1', 'doc2']
    assert fused[1]['doc_id'] in ['doc1', 'doc2']
    assert 'rrf_score' in fused[0]

if __name__ == "__main__":
    pytest.main([__file__])
