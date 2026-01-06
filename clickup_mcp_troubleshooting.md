# ClickUp MCP 서버 연결 문제 해결 과정

## 환경
- OS: Linux (Ubuntu)
- 도구: Codex CLI
- MCP Server: ClickUp (Smithery)

## 시도한 작업들

### 1. ClickUp MCP 서버 설치
```bash
npx -y @smithery/cli@latest install clickup --client codex
```
**결과:** ✅ 설치 성공

### 2. OAuth 인증 시도
```bash
codex mcp login clickup
```
**결과:** ❌ 인증 URL에서 에러 발생
- 브라우저에서 인증 URL 열었을 때 에러 발생 (구체적인 에러 내용 미상)

### 3. API 토큰 직접 설정 시도
ClickUp에서 직접 API 토큰을 발급받아 환경변수로 설정:
```bash
export CLICKUP_API_TOKEN="pk_..."
```

### 4. config.toml에 환경변수 설정 추가
`~/.codex/config.toml` 파일에 다음 설정 추가:
```toml
[mcp_servers.clickup]
url = "https://server.smithery.ai/clickup/mcp"

[mcp_servers.clickup.env]
CLICKUP_API_TOKEN = "${CLICKUP_API_TOKEN}"
```
**결과:** ❌ 에러 발생
```
Error: env is not supported for streamable_http
in `mcp_servers.clickup`
```

## 문제 분석

### 1. MCP 서버 타입 차이
- **Local MCP Server (stdio)**: `command` + `args` 사용, 환경변수 지원 ✅
- **Remote MCP Server (streamable_http)**: `url` 사용, 환경변수 미지원 ❌

ClickUp Smithery 서버는 `streamable_http` 타입으로:
- URL 기반: `https://server.smithery.ai/clickup/mcp`
- OAuth 인증 방식 사용
- 환경변수(`env`), 헤더(`headers`) 설정 불가

### 2. OAuth 인증 실패 원인 (추정)
- Smithery OAuth 플로우의 redirect 문제
- 로컬 콜백 서버(127.0.0.1:41415) 접근 문제
- 브라우저 인증 페이지 에러

### 3. API 토큰 직접 사용 불가
- Smithery 서버는 OAuth만 지원
- API 토큰을 직접 전달할 방법 없음
- 환경변수, 헤더 설정 모두 streamable_http에서 미지원

## 현재 상태

### config.toml 설정
```toml
[mcp_servers.clickup]
url = "https://server.smithery.ai/clickup/mcp"
startup_timeout_sec = 120.0
tool_timeout_sec = 300.0
```

### MCP 서버 목록
```
Name     Url                                     Status   Auth
clickup  https://server.smithery.ai/clickup/mcp  enabled  Unsupported
```

## 필요한 정보

1. **OAuth 인증 에러의 구체적인 내용**
   - 브라우저에 표시된 정확한 에러 메시지
   - 네트워크 탭의 실패한 요청 정보

2. **대안 확인 필요**
   - ClickUp 공식 MCP 서버 존재 여부
   - API 토큰 기반 로컬 MCP 서버 구현 가능성
   - Smithery OAuth 인증 우회 방법

## 검색 결과 및 원인 분석

### 핵심 발견 사항

#### 1. Smithery ClickUp MCP는 OAuth만 지원
- Smithery의 ClickUp MCP 서버(`https://server.smithery.ai/clickup/mcp`)는 **OAuth 2.1 with PKCE**만 지원
- API 토큰이나 Auth 액세스 토큰은 **사용 불가**
- streamable_http 타입이라 환경변수나 헤더 설정 불가

#### 2. Codex OAuth 구현에 알려진 문제
- GitHub 이슈: [Failing to login with OAuth in MCP HTTP server #5045](https://github.com/openai/codex/issues/5045)
- GitHub 이슈: [Attempts OAuth even when the MCP SSE server does not support oauth #5588](https://github.com/openai/codex/issues/5588)
- Codex의 OAuth 토큰 교환 실패: "Server returned error response: Required argument is missing"
- MCP SSE 서버가 OAuth를 지원하지 않아도 OAuth 시도하는 버그

#### 3. Smithery 설명
- "Client doesn't support OAuth yet or link isn't working?" 메시지 존재
- 대안으로 API 키를 포함한 URL 제공 옵션 있음

## 해결 방법

### ✅ 권장 해결책: API 토큰 지원하는 대안 서버 사용

Smithery 서버 대신, API 토큰을 지원하는 커뮤니티 제작 ClickUp MCP 서버 사용:

#### 옵션 1: @nazruden/clickup-mcp-server (권장)
- **타입**: stdio (로컬 실행)
- **인증**: CLICKUP_PERSONAL_TOKEN 환경변수
- **설치**: `npx @nazruden/clickup-server`
- **출처**: [GitHub - Nazruden/clickup-mcp-server](https://github.com/Nazruden/clickup-mcp-server)

**config.toml 설정 예시:**
```toml
[mcp_servers.clickup]
command = "npx"
args = ["-y", "@nazruden/clickup-server"]
startup_timeout_sec = 120.0
tool_timeout_sec = 300.0

[mcp_servers.clickup.env]
CLICKUP_PERSONAL_TOKEN = "${CLICKUP_PERSONAL_TOKEN}"
```

#### 옵션 2: @chykalophia/clickup-mcp-server
- **타입**: stdio
- **인증**: CLICKUP_API_TOKEN 환경변수
- **출처**: [npm - @chykalophia/clickup-mcp-server](https://www.npmjs.com/package/@chykalophia/clickup-mcp-server)

### ClickUp Personal API Token 발급 방법
1. ClickUp 로그인
2. **Settings** > **My Settings** > **Apps**
3. **API Token** 섹션에서 **Generate** 클릭
4. 토큰 복사 (형식: `pk_...`)

### 환경변수 설정
```bash
# .bashrc 또는 .zshrc에 추가
echo 'export CLICKUP_PERSONAL_TOKEN="pk_여기에_발급받은_토큰"' >> ~/.bashrc
source ~/.bashrc
```

## 참고 자료

### 공식 문서
- [ClickUp's MCP Server](https://developer.clickup.com/docs/connect-an-ai-assistant-to-clickups-mcp-server)
- [ClickUp Authentication](https://developer.clickup.com/docs/authentication)
- [What is ClickUp MCP?](https://help.clickup.com/hc/en-us/articles/33335772678423-What-is-ClickUp-MCP)

### GitHub 이슈
- [Failing to login with OAuth in MCP HTTP server #5045](https://github.com/openai/codex/issues/5045)
- [Attempts OAuth even when the MCP SSE server does not support oauth #5588](https://github.com/openai/codex/issues/5588)

### 대안 서버
- [Nazruden ClickUp MCP Server](https://github.com/Nazruden/clickup-mcp-server)
- [ClickUp MCP Server - npm](https://www.npmjs.com/package/@chykalophia/clickup-mcp-server)
- [Smithery - ClickUp MCP Server (Python)](https://smithery.ai/server/@Polaralias/clickup-mcp)

### 추가 정보
- [Connect to MCPs - Smithery Documentation](https://smithery.ai/docs/use/connect)
- [ClickUp MCP Server | MCP Servers · LobeHub](https://lobehub.com/mcp/nazruden-clickup-mcp-server)
