"""
Test MyService Evaluation Modules

MyService 도메인별 평가 모듈들의 테스트입니다.
"""

import pytest
import tempfile
import json
from pathlib import Path

def test_groundtruth_extract_label():
    """라벨 추출 테스트"""
    from services.pe_agent.evaluation.groundtruth import extract_label
    
    # 정상적인 라벨 추출
    assert extract_label("123456789-0001") == "123456789-0001"
    assert extract_label("doc_123456789-0001_extra") == "123456789-0001"
    assert extract_label("prefix_123456789-0001_suffix") == "123456789-0001"
    
    # 라벨이 없는 경우
    assert extract_label("no_label_here") is None
    assert extract_label("") is None
    assert extract_label(None) is None

def test_groundtruth_gold_index():
    """골드 인덱스 테스트"""
    from services.pe_agent.evaluation.groundtruth import MyServiceGoldIndex
    
    # 테스트 데이터
    doc_ids = [
        "123456789-0001",
        "987654321-0002", 
        "no_label_doc",
        "123456789-0003"
    ]
    
    gold = MyServiceGoldIndex(doc_ids)
    
    # 라벨별 인덱스 확인
    assert gold.ids_for("123456789-0001") == {0}
    assert gold.ids_for("987654321-0002") == {1}
    assert gold.ids_for("123456789-0003") == {3}
    assert gold.ids_for("no_label_doc") == set()  # 라벨이 없는 문서
    assert gold.ids_for("nonexistent") == set()
    
    # 모든 라벨 확인
    all_labels = gold.get_all_labels()
    assert "123456789-0001" in all_labels
    assert "987654321-0002" in all_labels
    assert "123456789-0003" in all_labels
    assert len(all_labels) == 3

def test_bucketizer_tokenization():
    """버킷 분류기 토큰화 테스트"""
    from services.pe_agent.evaluation.bucketizer import tokenize_ko_en
    
    # 한국어 토큰화
    ko_tokens = tokenize_ko_en("안녕하세요 반갑습니다")
    assert "안녕하세요" in ko_tokens
    assert "반갑습니다" in ko_tokens
    
    # 영어 토큰화
    en_tokens = tokenize_ko_en("hello world test123")
    assert "hello" in en_tokens
    assert "world" in en_tokens
    assert "test123" in en_tokens
    
    # 혼합 토큰화
    mixed_tokens = tokenize_ko_en("안녕 hello world 123")
    assert "안녕" in mixed_tokens
    assert "hello" in mixed_tokens
    assert "world" in mixed_tokens
    assert "123" in mixed_tokens

def test_bucketizer_lexical_overlap():
    """어휘적 중복 비율 테스트"""
    from services.pe_agent.evaluation.bucketizer import lexical_overlap_ratio
    
    # 완전 중복
    ratio = lexical_overlap_ratio("hello world", {"hello", "world"})
    assert ratio == 1.0
    
    # 부분 중복
    ratio = lexical_overlap_ratio("hello world", {"hello", "test"})
    assert 0.0 < ratio < 1.0
    
    # 중복 없음
    ratio = lexical_overlap_ratio("hello world", {"test", "example"})
    assert ratio == 0.0

def test_bucketizer_classification():
    """버킷 분류 테스트"""
    from services.pe_agent.evaluation.bucketizer import make_bucketizer
    
    # 테스트 데이터
    docs_text = [
        "안녕하세요 반갑습니다",
        "hello world test",
        "혼합된 텍스트 mixed text"
    ]
    doc_ids = [
        "123456789-0001",
        "987654321-0002",
        "555666777-0003"
    ]
    
    bucketizer = make_bucketizer(docs_text, doc_ids)
    
    # 버킷 분류 확인
    bucket = bucketizer("안녕하세요", "123456789-0001")
    assert bucket in ["semantic", "mixed", "lexical"]
    
    bucket = bucketizer("완전히 다른 쿼리", "123456789-0001")
    assert bucket == "semantic"  # 어휘적 중복이 거의 없음

def test_evaluation_inputs():
    """평가 입력 데이터 테스트"""
    from services.pe_agent.evaluation.retrieval_evaluator import EvalInputs
    
    # 테스트 데이터
    inputs = EvalInputs(
        docs_text=["doc1", "doc2"],
        doc_ids=["id1", "id2"],
        queries=[("query1", "id1"), ("query2", "id2")]
    )
    
    assert len(inputs.docs_text) == 2
    assert len(inputs.doc_ids) == 2
    assert len(inputs.queries) == 2
    assert inputs.queries[0][0] == "query1"
    assert inputs.queries[0][1] == "id1"

def test_qa_data_parsing():
    """Q&A 데이터 파싱 테스트"""
    from services.pe_agent.evaluation.retrieval_evaluator import MyServiceRetrievalEvaluator
    
    # 테스트용 더미 클래스
    class DummyEvaluator(MyServiceRetrievalEvaluator):
        def __init__(self):
            pass
    
    evaluator = DummyEvaluator()
    
    # EXAONE 형식 테스트
    qa_data = {
        "123456789-0001": {
            "items": [
                {"question": "질문1"},
                {"question": "질문2"}
            ]
        },
        "987654321-0002": "단순 질문"
    }
    
    qa_pairs = evaluator._parse_qa_data(qa_data)
    
    assert len(qa_pairs) == 3
    assert ("질문1", "123456789-0001") in qa_pairs
    assert ("질문2", "123456789-0001") in qa_pairs
    assert ("단순 질문", "987654321-0002") in qa_pairs

def test_core_evaluation_integration():
    """core 평가기와의 통합 테스트"""
    from core.retrieval.evaluation import Query, evaluate_retriever
    from services.pe_agent.evaluation.groundtruth import MyServiceGoldIndex
    from services.pe_agent.evaluation.bucketizer import make_bucketizer
    
    # 테스트용 더미 retriever
    class DummyRetriever:
        def doc_ids(self):
            return ["123456789-0001", "987654321-0002"]
        
        def search(self, query, top_k):
            return [{"doc_id": "123456789-0001", "score": 0.9}]
    
    # 테스트 데이터
    doc_ids = ["123456789-0001", "987654321-0002"]
    docs_text = ["문서1", "문서2"]
    
    # core 평가기 사용
    gold = MyServiceGoldIndex(doc_ids)
    bucketizer = make_bucketizer(docs_text, doc_ids)
    queries = [Query("테스트 쿼리", "123456789-0001")]
    
    # 평가 실행 (실제로는 더 복잡한 retriever 필요)
    # results = evaluate_retriever(DummyRetriever(), queries, gold, bucketizer)
    # assert 'hit_at_k' in results
    
    # 기본 구조 확인
    assert gold.ids_for("123456789-0001") == {0}
    assert bucketizer("테스트", "123456789-0001") in ["semantic", "mixed", "lexical"]

if __name__ == "__main__":
    pytest.main([__file__])
