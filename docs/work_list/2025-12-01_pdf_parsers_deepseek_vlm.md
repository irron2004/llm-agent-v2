# 2025-12-01 PDF 파서 엔진/어댑터 분리, DeepSeek VLM 추가, 설정화

## 작업 개요
- 기존 PDF 파서를 엔진/어댑터 구조로 분리하고, DeepSeek 기반 VLM 파서를 추가했습니다.
- DeepDoc/DeepSeek/Plain 파서 모두 공통 인터페이스(`ParsedDocument`, `PdfParseOptions`)로 통합했습니다.
- `.env` 기반으로 DeepDoc/DeepSeek 모델/프롬프트를 교체할 수 있게 설정을 추가했습니다.
- Ingest 서비스에서 DeepDoc/VLM 전용 팩토리 메서드를 제공해 선택 구성이 쉬워졌습니다.

## 변경 사항
- `llm_infrastructure/preprocessing/parsers/engines/`
  - `pdf_vlm_engine.py`: VLM 기반 파서 엔진 (`VlmPdfEngine`, DeepSeek-VL 등 주입)
  - `pdf_deepdoc_engine.py`, `pdf_plain_engine.py` 유지
- `llm_infrastructure/preprocessing/parsers/adapters/`
  - `pdf_vlm.py`: 레지스트리 등록 (`pdf_vlm`, 호환 `pdf_deepseek_vl`) / alias `DeepSeekVLPdfAdapter`
  - 기존 plain/deepdoc 어댑터 유지
- `llm_infrastructure/preprocessing/parsers/base.py`
  - `PdfParseOptions`에 VLM 필드 추가 (`vlm_model`, `vlm_prompt`, `vlm_max_new_tokens`, `vlm_temperature`)
- `llm_infrastructure/preprocessing/__init__.py`
  - VLM 어댑터/alias 공개
- `config/settings.py`
  - `DeepSeekSettings` 추가 (모델 id, 프롬프트, HF 엔드포인트, 모델 캐시 경로, 다운로드 허용, 디바이스)
  - 기존 `DeepDocSettings`와 함께 export
- `services/ingest/document_ingest_service.py`
  - `for_deepdoc()`, `for_vlm()` 팩토리 제공
  - 파서 id에 따라 DeepDoc/DeepSeek 기본 옵션 자동 적용, 커스텀 `vlm_client`/`vlm_factory`/`renderer` 주입 가능
- 테스트: `tests/preprocessing/test_pdf_parsers.py`에 VLM 경로 및 서비스 팩토리 커버리지 추가 (전체 통과)

## 사용법 요약
- DeepDoc 사용: `DocumentIngestService.for_deepdoc()`
- VLM/DeepSeek 사용:
  ```python
  def build_vlm(opts: PdfParseOptions):
      return MyVLMClient(model=opts.vlm_model, endpoint=opts.hf_endpoint)

  svc = DocumentIngestService.for_vlm(vlm_factory=build_vlm, renderer=my_pdf_to_images)
  result = svc.ingest_pdf(open("doc.pdf", "rb"))
  ```
- `.env` 예시:
  ```
  DEEPSEEK_MODEL_ID=deepseek-ai/deepseek-vl2
  DEEPSEEK_PROMPT="Extract all text as Markdown. Preserve tables and formulas."
  DEEPSEEK_HF_ENDPOINT=https://hf-mirror.com
  DEEPSEEK_MODEL_ROOT=./data/deepseek_models
  DEEPSEEK_ALLOW_DOWNLOAD=true
  DEEPSEEK_DEVICE=cpu
  ```

## 파일 경로
- 엔진: `backend/llm_infrastructure/preprocessing/parsers/engines/pdf_vlm_engine.py`
- 어댑터: `backend/llm_infrastructure/preprocessing/parsers/adapters/pdf_vlm.py`
- 서비스: `backend/services/ingest/document_ingest_service.py`
- 설정: `backend/config/settings.py`
- 테스트: `backend/tests/preprocessing/test_pdf_parsers.py`
