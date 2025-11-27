# 리팩토링이 필요한 이유 정리

RAGFlow 중심의 MVP에서 **내가 통제하는 RAG/LLM 플랫폼**으로 넘어가기 위해 진행한 리팩토링의 배경과 효과를 정리합니다. 각 항목은 과거 고통점 → 지금 구조의 이득을 1:1로 대응합니다.

---

## 1. RAGFlow UI에서 독립 → 화면 커스터마이징 지옥 탈출

**기존 문제:**
- ragflow 웹 코드를 `overrides-web` 식으로 복사해서 수정
- 권한 문제, 빌드/업데이트 시 파일이 덮여 커스텀 화면이 사라짐
- ragflow 버전 올릴 때마다 “머지 지옥”

**지금 구조에서의 이득:**
- FE를 아예 `frontend/`로 **독립 프로젝트**로 분리
- ragflow는 더 이상 UI 프레임워크가 아니라 **백엔드 서비스/파서** 역할
- ragflow 업데이트해도 내 FE 코드에는 영향 없음
- 권한/라우팅/상태관리를 내 프론트 기준으로 설계 가능 → public chat, 실험용 화면, 디버그용 JSON viewer 등을 자유롭게 추가/수정

> 한 줄 요약: **“ragflow 화면 위에 덧칠” → “내가 주인인 FE + ragflow는 단순 백엔드 자원”**

---

## 2. Retrieval 커스터마이징 자유도 폭증

**기존 문제:**
- ragflow 기본 hybrid retrieval 외에는 multi-query 조합, 다양한 reranking, 커스텀 스코어링을 넣기 어려움

**지금 구조에서의 이득:**
- `backend/llm_infrastructure/retrieval/engines/` + `adapters/` + `registry` 패턴으로 재구성
  - dense 전용, sparse 전용, hybrid, RRF, reranker 등을 **엔진 단위로 분리**
  - “프리셋(preset)”에서 조합을 설정 파일로 선택
- service layer (`services/search_service.py`, `chat_service.py`)에서 실험 조합을 쉽게 전환
  - 예: hybrid + RRF + multi-query 4개 / 운영: dense-only + reranker

> ragflow 캔버스 제약에서 벗어나 **retrieval 흐름이 코드 차원에서 내 손 안으로** 들어옴.

---

## 3. ElasticSearch/인덱싱 디버깅 가능

**기존 문제:**
- LLM이 ES 인덱스를 chunk별로 만들었지만
  - 어떤 경우는 인덱스가 안 만들어지거나
  - 만들어졌는데도 검색에 안 쓰임
- ragflow 내부 파이프라인이라 어디서 막히는지 추적 어려움

**지금 구조에서의 이득:**
- `backend`에서 **인덱스 빌드 API**를 직접 정의 (`/index/build`)
  - chunking → embedding → ES/로컬 인덱싱 전 과정을 내 코드 안에서 수행
- 인덱스 생성 결과(문서 수, 벡터 수, 에러 로그)를 내가 정의한 방식으로 로깅
- ES 사용 시 매핑/분석기/인덱스 이름까지 코드에서 명시적 제어, 실제 검색 사용 여부도 retriever 엔진 레벨에서 확인 가능

> **인덱싱/검색 흐름을 100% 추적 가능**해져 “어디서 안 물리는지 모르겠다” 문제 해소.

---

## 4. 데이터 정규화(L0~L5) + 사전(variant map) 관리

**기존 문제:**
- 도메인 사전(반도체 용어 등) 정규화를 ragflow 파이프라인에 억지로 끼워 넣다 보니 코드 꼬임
- 어떤 레벨의 정규화가 어디에서 적용되는지 관리 어려움

**지금 구조에서의 이득:**
- `llm_infrastructure/preprocessing/normalize_engine/`에 L0~L5 정규화 엔진을 **명시적으로 분리**
  - L0: 기본 / L1: variant 맵 / L3~L5: 도메인 특화 규칙…
- `adapters/normalize.py` + `registry.py`로 `get_preprocessor("normalize", level="L3")` 식 호출
- 사전(variant map)은 엔진 혹은 별도 rules 모듈에서 중앙 관리
- 운영/실험에서 프리셋만 바꿔 레벨 스위칭 가능 (예: 운영 L1, 실험 L3, 향후 L4/L5 추가)

> **사전 기반 정규화가 ragflow 내부에서 꼬이는 대신, 내 엔진+프리셋 구조로 정리**되어 재사용/실험이 쉬움.

---

## 5. env/설정 관리: “설정 바꾸려고 빌드” 지옥 탈출

**기존 문제:**
- ragflow 환경에서 `.env` 관리가 깔끔하지 않아 설정 변경 시 docker 이미지를 다시 빌드해야 했고, 빌드 시간 때문에 실험/배포 사이클이 느림

**지금 구조에서의 이득:**
- `backend/config/settings.py` + `.env` + Pydantic Settings로 **환경 변수만으로** 모델/임베딩/정규화 레벨/preset 등을 변경
- Docker 이미지는 코드+라이브러리만 포함 → `.env`/config 변경은 컨테이너 재시작만으로 반영
- 확장: 활성 preset을 DB/Redis에 두면 서버 재시작 없이 런타임 스위칭(shadow test, A/B)도 가능

> **코드 수정 = 빌드 / 설정 변경 = 재시작 또는 런타임 스위치**로 분리되어 실험 속도 향상.

---

## 6. Multi-Query(MQ) 제대로 구현

**기존 문제:**
- multi-agent로 MQ를 만들었지만 내부 한계로 `"[query1, query2, query3]"` 문자열 한 덩어리만 검색되는 등 반복 검색 로직 삽입이 어려움

**지금 구조에서의 이득:**
- `retrieval/engines/`에 Multi-Query + RRF를 직접 구현
  - Query 리스트를 실제 Python list로 다루고 query별 search → RRF merge 수행
- `retrieval preset`에 `multi_query.enabled`, `n`, `include_original` 등을 옵션으로 넣어 실험
- 동의어 기반 MQ, LLM 기반 query decomposition 등도 엔진/서비스 레벨에서 직접 설계 가능

> **문자열로 깨지는 MQ가 사라지고, 논문/아이디어대로 MQ를 구현**할 수 있음.

---

## 7. Tool calling / vLLM / GPT-OSS-20B 연동 자유도

**기존 문제:**
- vLLM, GPT-OSS-20B 등이 지원하는 structured output/함수 호출 패턴을 ragflow 에이전트 구조에서 자연스럽게 붙이기 어려움

**지금 구조에서의 이득:**
- `llm_infrastructure/llm/engines/`에서 vLLM, gpt-oss-20b, OpenAI compatible API를 한 곳에서 관리
- service layer에서 답변 생성/툴 호출 JSON 생성/백엔드 툴 실행/후속 호출 루프를 자유롭게 설계
- LangGraph/LangChain/custom agent 루프를 붙여도 ragflow 캔버스 제약 없이 **코드 레벨에서 control loop**를 설계 가능

> 모델이 지원하는 기능(툴 호출, 구조화 출력)을 **ragflow 한계 없이 풀로 활용**.

---

## 8. 실험 관리/로그/프리셋: “연구 + 서비스” 목표 적합

**목표 상기:** “여러 방법론/논문을 서비스에 적용해 보고, 다양한 실험을 관리할 수 있는 구조” 만들기.

**지금 구조에서의 이득:**
- **프리셋(preset):** 하나의 실험 설정 = 하나의 preset 파일(정규화 레벨, embedding, retrieval, MQ, reranker…)
- **레지스트리(registry):** 새 방법론을 plugin처럼 추가 후 이름만 preset에 기재
- **service layer:** 한 번의 API 호출 안에서 “어떤 preset으로 어떤 실험을 돌렸는지” 로그 남기기 용이
- **로그 구조:** query, preset_id, retrieval 결과, answer, latency 등을 기록해 비교/분석 가능

> ragflow에 “맞춰서 겨우 돌리는 서비스”가 아니라, **논문/아이디어를 빠르게 꽂아볼 수 있는 나만의 RAG/LLM 실험 플랫폼**으로 진화.

---

## 한 줄 대응표: 과거 고통 → 현재 구조

1. **UI:** ragflow override → 내 FE 프로젝트 (독립, 안정)
2. **Retrieval 커스텀:** ragflow hybrid 한계 → pluggable 엔진 + preset
3. **ES 인덱스:** 내부 블랙박스 → 내 코드로 인덱스/검색 제어 및 디버깅
4. **정규화/사전:** ragflow 안 꼬임 → L0~L5 normalize 엔진 + variant map 중앙 관리
5. **설정/빌드:** 설정 바꾸려면 빌드 → env/preset 스위치(재시작 or 런타임)
6. **Multi-Query:** 문자열로 깨짐 → 진짜 MQ + RRF 구현 가능
7. **Tool calling:** ragflow 한계 → vLLM/OSS 모델 기능을 서비스 로직에서 직접 활용

---

## 다음 액션 아이디어
- 가장 먼저 체감하고 싶은 장점(예: MQ, 정규화, 인덱스 디버깅 등)을 하나 고르고, 그 기준으로 1~2주짜리 작은 마일스톤 플랜 수립
