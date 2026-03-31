# Git 브랜치 워크플로우

## 브랜치 구조

| 브랜치 | 역할 | 배포 명령 | 포트 |
|--------|------|-----------|------|
| `main` | prod (사용자 서비스) | `make prod-up` | 8001 / 9097 |
| `dev` | 통합/테스트 | `make dev-up` | 8011 / 9098 |
| `feat/xxx` | 개별 작업 | — | — |

---

## 기본 워크플로우

### 1. 작업 시작

```bash
make feat name=<작업명>
# 예: make feat name=reranker-upgrade
# → dev 기준으로 feat/reranker-upgrade 브랜치 생성 후 자동 전환
```

### 2. 작업 및 커밋

```bash
# 코드 수정 후
git add <파일>
git commit -m "feat: 설명"
```

### 3. dev에 머지 후 테스트

```bash
git checkout dev
git merge feat/<작업명>
make dev-up        # 포트 9098에서 테스트
```

### 4. 확정 후 prod 배포

```bash
git checkout main
git merge dev
make prod-up       # 미커밋 변경사항 있으면 자동 차단
```

---

## 규칙

- **`main`에 직접 커밋하지 않는다.** 항상 `feat/` → `dev` → `main` 순서로.
- **`make prod-up` 전에 반드시 커밋한다.** 미커밋 변경사항이 있으면 배포가 차단됨.
- **feature 브랜치는 `dev` 기준으로 만든다.** (`make feat` 가 자동 처리)

---

## Makefile 명령어 요약

| 명령어 | 동작 |
|--------|------|
| `make feat name=xxx` | `dev` 기준으로 `feat/xxx` 브랜치 생성 |
| `make dev-up` | `dev` 브랜치로 전환 후 dev 컨테이너 실행 |
| `make dev-rebuild` | `dev` 브랜치로 전환 후 dev 컨테이너 재빌드 |
| `make prod-up` | 미커밋 체크 → `main` 브랜치로 전환 후 prod 배포 |
