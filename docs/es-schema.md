# Elasticsearch Index Schema: `rag_chunks_dev_v2`

## Overview

RAG 검색용 Elasticsearch 인덱스. 반도체 장비 문서를 청킹하여 벡터 임베딩과 함께 저장한다.

- **총 문서 수:** 345,935
- **임베딩 모델:** BAAI/bge-base-en-v1.5 (768차원)
- **청킹:** fixed_size (512 tokens, overlap 50)
- **텍스트 분석기:** Nori (한국어 형태소 분석)

## Index Metadata (`_meta`)

```json
{
  "pipeline": {
    "index_purpose": "rag_retrieval",
    "preprocess": { "method": "normalize" },
    "chunking": { "method": "fixed_size", "size": 512, "overlap": 50 },
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "embedding_dim": 768
  }
}
```

## 필드 목록

### 식별 필드

| 필드 | 타입 | 검색 가능 | 설명 |
|------|------|-----------|------|
| `doc_id` | keyword | O | 원본 문서 ID |
| `chunk_id` | keyword | O | 청크 고유 ID (예: `40059601#0000`) |
| `content_hash` | keyword | O | 콘텐츠 해시 (중복 검출용) |

### 장비 메타데이터

| 필드 | 타입 | 검색 가능 | 설명 |
|------|------|-----------|------|
| `device_name` | keyword | O | 장비명 (예: DAS2000, SUPRA N) |
| `equip_id` | keyword | O | 장비 모델 ID (예: EPAG50) |
| `doc_type` | keyword | O | 문서 유형 (아래 분포 참조) |

### 텍스트 콘텐츠

| 필드 | 타입 | 분석기 | 설명 |
|------|------|--------|------|
| `content` | text | nori | 청크 본문 텍스트 |
| `search_text` | text | nori | 검색 최적화 텍스트 (메타데이터 포함) |
| `caption` | text | (비검색) | 이미지/테이블 캡션 |
| `summary` | text | (비검색) | 문서 요약 |
| `chunk_summary` | text | standard | 청크 단위 요약 |
| `doc_description` | text | (비검색) | 문서 설명 |

### 벡터 임베딩

| 필드 | 타입 | 설명 |
|------|------|------|
| `embedding` | dense_vector (768d) | BGE-base-en-v1.5 임베딩. cosine 유사도, int8_hnsw 인덱스 (m=16, ef_construction=100) |

### 구조 정보

| 필드 | 타입 | 검색 가능 | 설명 |
|------|------|-----------|------|
| `page` | integer | O | 원본 문서 페이지 번호 |
| `chapter` | keyword | O | 챕터/섹션명 |
| `page_image_path` | keyword | X | 페이지 이미지 경로 |
| `bbox` | object | X | 바운딩 박스 (비활성) |

### 분류 및 태그

| 필드 | 타입 | 설명 |
|------|------|------|
| `tags` | keyword | 태그 배열 (예: `["myservice", "batch"]`) |
| `chunk_keywords` | keyword + text(sub) | 청크 키워드 배열 (keyword + standard 분석기 서브필드) |
| `lang` | keyword | 언어 코드 (예: `ko`, `en`) |

### 관리 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `created_at` | date | 생성 시각 |
| `updated_at` | date | 수정 시각 |
| `pipeline_version` | keyword | 전처리 파이프라인 버전 (예: `v1`) |
| `tenant_id` | keyword | 테넌트 ID |
| `project_id` | keyword | 프로젝트 ID |
| `quality_score` | float | 청크 품질 점수 |

## Nori 분석기 설정

```json
{
  "analyzer": {
    "nori": {
      "type": "custom",
      "tokenizer": "nori_tokenizer",
      "filter": ["nori_readingform", "lowercase"]
    }
  }
}
```

`content`와 `search_text` 필드에 적용. 한국어 형태소 분석 + 한자 읽기 변환 + 소문자화.

## doc_type 분포

| doc_type | 문서 수 | 비율 |
|----------|---------|------|
| myservice | 324,482 | 93.8% |
| SOP | 11,452 | 3.3% |
| gcb | 7,264 | 2.1% |
| set_up_manual | 2,045 | 0.6% |
| trouble_shooting_guide | 688 | 0.2% |
| sample | 4 | <0.1% |

## 샘플 문서

```json
{
  "doc_id": "40059601",
  "chunk_id": "40059601#0000",
  "page": 0,
  "content": "FM to CTC communication alarm",
  "search_text": "status FM to CTC communication alarm myservice batch",
  "lang": "ko",
  "doc_type": "myservice",
  "device_name": "DAS2000",
  "doc_description": "CRMicro EAFP12 Robot communication alarm",
  "chapter": "status",
  "chunk_summary": "FM to CTC communication alarm observed.",
  "chunk_keywords": ["FM", "CTC", "communication alarm"],
  "tags": ["myservice", "batch"],
  "pipeline_version": "v1",
  "created_at": "2025-12-31T03:02:38.596835+00:00",
  "updated_at": "2025-12-31T03:02:38.596835+00:00"
}
```
