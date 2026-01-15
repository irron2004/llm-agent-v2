# 참고 문서 이미지 표시(페이지 이미지 + bbox 하이라이트) 업무지시서

## 목표
- 참고 문서를 텍스트가 아니라 페이지 이미지로 표시한다.
- ES에 이미지 URL을 저장하지 않고 doc_id + page 규칙으로 URL을 계산한다.
- 저장소는 온프레미스 MinIO(S3 호환)를 사용하며 외부 공개는 하지 않는다.
- 이후 bbox 하이라이트를 적용할 수 있도록 확장 경로를 마련한다.

---

## 핵심 결정사항
- 이미지 저장 경로 규칙: `documents/{doc_id}/page_{page}.png`
- 버킷명: `doc-images`
- URL 생성 방식: API 응답에서 doc_id + page로 계산
- 접근 방식: 브라우저가 MinIO에 직접 접근하지 않고 백엔드 프록시 엔드포인트를 통해 이미지 제공

---

## 작업 범위
1) MinIO 서비스 추가 (docker-compose)
2) MinIO 업로드 클라이언트 및 설정 추가
3) PDF→이미지 렌더링 단계에서 MinIO 저장
4) 검색 응답에 이미지 URL 포함(계산 방식)
5) 프런트 참고 문서 UI에서 이미지 렌더링
6) bbox 하이라이트를 위한 후속 설계(좌표 생성 파이프라인)

---

## 1) docker-compose에 MinIO 추가
### 1.1 서비스 추가
- `docker-compose.yml`에 `minio` 서비스 추가
- 내부망 전용이므로 포트 노출은 최소화
  - 콘솔/관리 필요 시에만 `127.0.0.1` 바인딩으로 제한

예시(설치 지침용, 실제 설정 시 값 확인):
```
  minio:
    container_name: rag-minio
    image: minio/minio:RELEASE.2024-12-18T02-33-49Z
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=${MINIO_ROOT_USER}
      - MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}
    volumes:
      - ./data/minio:/data
    networks:
      - rag_net
    # 내부망 전용. 콘솔 필요 시만 노출:
    # ports:
    #   - "127.0.0.1:9000:9000"
    #   - "127.0.0.1:9001:9001"
    restart: unless-stopped

  minio-init:
    image: minio/mc
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      /usr/bin/mc alias set local http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} &&
      /usr/bin/mc mb -p local/doc-images || true
      "
    networks:
      - rag_net
```

### 1.2 .env 변수 추가
- `.env`에 아래 값 추가(실제 값은 보안 정책에 맞게)
```
MINIO_ROOT_USER=...
MINIO_ROOT_PASSWORD=...
MINIO_BUCKET=doc-images
MINIO_ENDPOINT=http://minio:9000
MINIO_SECURE=false
```

---

## 2) MinIO 업로드 클라이언트 추가
### 2.1 라이브러리 추가
```bash
pip install minio
```
- `pyproject.toml`/`uv.lock`에 반영

### 2.2 설정 및 유틸 구현
Update file: `backend/config/settings.py`
- `MinioSettings` 추가 (env prefix: `MINIO_`)
- 필드: `endpoint`, `access_key`, `secret_key`, `bucket`, `secure`
- 전역 설정 객체로 노출

New file: `backend/services/storage/minio_client.py`
- `get_minio_client()`, `ensure_bucket()`, `upload_bytes()`
- `upload_bytes`는 `object_name`과 `content_type`을 받아 업로드

---

## 3) PDF→이미지 렌더링 시 MinIO 저장
Update file: `backend/services/es_ingest_service.py`
- `ingest_pdf()`에서 `DocumentIngestService.for_vlm()` 호출 시 renderer 주입
- doc_id가 다르므로 DocumentIngestService를 문서마다 생성하거나 renderer를 doc_id별로 생성

### 3.1 렌더러 동작 요건
- 입력: `pdf_bytes`
- 출력: 페이지 이미지 리스트(기존 VLM 파서와 동일)
- 동작:
  1) `pdf2image.convert_from_bytes`로 페이지 이미지 생성
  2) 각 페이지 이미지 PNG로 인코딩
  3) MinIO에 업로드(`documents/{doc_id}/page_{page}.png`)
  4) 이미지 리스트 반환(VLM 파싱은 그대로 진행)

### 3.2 doc_id 안전성
- `doc_id`에 `/` 또는 공백이 있을 수 있으니 경로에 쓸 때는 정규화/URL 인코딩 적용
- 권장: `safe_doc_id = re.sub(r"[^a-zA-Z0-9._-]", "_", doc_id)`

---

## 4) 이미지 URL 계산 및 API 응답 확장
Update file: `backend/api/routers/chat.py`
- `RetrievedDoc`에 `page: int | None`, `page_image_url: str | None` 추가
- `_to_retrieved_docs`에서 `page = result.metadata.get("page")` 추출
- `page_image_url = f"/api/assets/docs/{doc_id}/pages/{page}"`

Update file: `backend/api/routers/agent.py`
- 위와 동일하게 `RetrievedDoc`와 `_to_retrieved_docs` 수정

New file: `backend/api/routers/assets.py`
- `GET /assets/docs/{doc_id}/pages/{page}`
- MinIO에서 오브젝트를 읽어 `image/png`로 스트리밍
- 404/500 처리 및 `Cache-Control` 추가 권장

---

## 5) 프런트 참고 문서 UI 변경
Update file: `frontend/src/features/chat/types.ts`
- `RetrievedDoc` 타입에 `page?: number`, `pageImageUrl?: string` 추가

UI 변경 포인트
- 참조 문서 리스트에서 `pageImageUrl`이 있으면 이미지로 렌더링
- 없으면 기존 텍스트 스니펫 표시
- 이미지 로딩 실패 시 fallback 텍스트 표시

예시 렌더링
```
{doc.pageImageUrl ? (
  <div className="reference-image-wrapper">
    <img src={doc.pageImageUrl} alt={doc.title} />
    <!-- bbox overlay는 추후 -->
  </div>
) : (
  <div className="reference-text">{doc.snippet}</div>
)}
```

---

## 6) bbox 하이라이트(후속 단계)
Update file: `backend/llm_infrastructure/elasticsearch/mappings.py`
- `bbox`는 단일 객체 기준
- 여러 박스 필요 시 `bboxes` 필드(배열) 추가 고려

bbox 생성 경로(선택)
1) DeepDoc 파서 전환: OCR/레이아웃에서 bbox 획득
2) VLM 유지 + 별도 OCR/레이아웃 단계: bbox만 추출

좌표 권장 포맷
- 정규화 좌표: `{x0: 0~1, y0: 0~1, x1: 0~1, y1: 0~1}`
- 프런트에서 이미지 실제 크기에 맞춰 스케일링

---

## 검증 체크리스트
- [ ] MinIO 컨테이너 정상 기동 및 버킷 생성 확인
- [ ] PDF ingest 시 MinIO에 페이지 이미지 저장 확인
- [ ] `GET /api/assets/docs/{doc_id}/pages/{page}` 응답 확인
- [ ] 검색 응답에 `page` 및 `page_image_url` 포함
- [ ] 프런트에서 참고 문서가 이미지로 표시되는지 확인

---

## 참고 파일
- `backend/services/es_ingest_service.py`
- `backend/services/ingest/document_ingest_service.py`
- `backend/llm_infrastructure/preprocessing/parsers/engines/pdf_vlm_engine.py`
- `backend/llm_infrastructure/elasticsearch/document.py`
- `backend/api/routers/chat.py`
- `backend/api/routers/agent.py`
- `frontend/src/features/chat/types.ts`
