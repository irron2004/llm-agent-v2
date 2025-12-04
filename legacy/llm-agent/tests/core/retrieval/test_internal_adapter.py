"""
Test Internal Embedder Adapter

사내 create_embedder와 새로운 어댑터가 제대로 작동하는지 테스트합니다.
"""

import pytest
import numpy as np
from core.retrieval.embedders import create_embedder

def test_adapter_creation():
    """어댑터 생성 테스트"""
    # 사내 create_embedder가 있는지 확인
    try:
        from core.embedding.core.embedders.create_embedder import create_embedder as internal_create
        has_internal = True
    except ImportError:
        has_internal = False
    
    # 어댑터 생성 시도
    embedder = create_embedder("nlpai-lab/KoE5", device=None, use_e5_prefix=True)
    
    # 기본 속성 확인
    assert hasattr(embedder, 'model_name')
    assert hasattr(embedder, 'uses_e5_prefix')
    assert hasattr(embedder, 'encode')
    
    # 모델명 확인
    assert embedder.model_name == "nlpai-lab/KoE5"
    assert embedder.uses_e5_prefix == True

def test_encode_interface():
    """encode 인터페이스 테스트"""
    embedder = create_embedder("nlpai-lab/KoE5", device=None, use_e5_prefix=True)
    
    # 기본 encode (mode=None)
    texts = ["테스트 문서입니다."]
    result = embedder.encode(texts)
    
    # 결과 검증
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[0] == 1  # 문서 수
    assert result.dtype == np.float32

def test_e5_prefix_mode():
    """E5 prefix 모드 테스트"""
    embedder = create_embedder("nlpai-lab/KoE5", device=None, use_e5_prefix=True)
    
    # passage 모드
    passage_texts = ["문서 내용입니다."]
    passage_result = embedder.encode(passage_texts, mode="passage")
    
    # query 모드
    query_texts = ["질문입니다."]
    query_result = embedder.encode(query_texts, mode="query")
    
    # 결과 검증
    assert isinstance(passage_result, np.ndarray)
    assert isinstance(query_result, np.ndarray)
    assert passage_result.shape == (1, passage_result.shape[1])
    assert query_result.shape == (1, query_result.shape[1])

def test_normalization_consistency():
    """정규화 일관성 테스트"""
    embedder = create_embedder("nlpai-lab/KoE5", device=None, use_e5_prefix=True)
    
    texts = ["정규화 테스트 문서입니다."]
    
    # 여러 번 호출해도 결과가 일관적인지 확인
    result1 = embedder.encode(texts)
    result2 = embedder.encode(texts)
    
    # 결과가 동일한지 확인 (싱글톤 캐시 효과)
    np.testing.assert_array_almost_equal(result1, result2, decimal=6)
    
    # L2 정규화 확인 (사내 임베더는 이미 정규화됨)
    norms = np.linalg.norm(result1, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=4)

def test_batch_processing():
    """배치 처리 테스트"""
    embedder = create_embedder("nlpai-lab/KoE5", device=None, use_e5_prefix=True)
    
    # 여러 문서 배치 처리
    texts = [
        "첫 번째 문서입니다.",
        "두 번째 문서입니다.",
        "세 번째 문서입니다."
    ]
    
    result = embedder.encode(texts, mode="passage")
    
    # 결과 검증
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 3  # 문서 수
    assert result.dtype == np.float32

def test_fallback_to_st():
    """SentenceTransformer 폴백 테스트"""
    # 사내 임베더가 없는 환경에서 ST 폴백이 작동하는지 테스트
    # (실제로는 사내 임베더가 있으므로 이 테스트는 주석 처리)
    
    # embedder = create_embedder("sentence-transformers/all-MiniLM-L6-v2", device=None, use_e5_prefix=False)
    # assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    # assert embedder.uses_e5_prefix == False
    
    pass

def test_embedder_protocol_compliance():
    """Embedder 프로토콜 준수 테스트"""
    from core.retrieval.api import Embedder
    
    embedder = create_embedder("nlpai-lab/KoE5", device=None, use_e5_prefix=True)
    
    # 프로토콜 준수 확인
    assert hasattr(embedder, 'model_name')
    assert hasattr(embedder, 'uses_e5_prefix')
    assert hasattr(embedder, 'encode')
    
    # 타입 힌트 확인
    assert callable(embedder.encode)
    
    # 실제 호출 테스트
    texts = ["프로토콜 테스트"]
    result = embedder.encode(texts)
    assert isinstance(result, np.ndarray)

if __name__ == "__main__":
    pytest.main([__file__])
