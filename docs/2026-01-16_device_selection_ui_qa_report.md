# 기기 선택 UI 스크롤 문제 QA 보고서

**작성일**: 2026-01-16  
**작성자**: AI Assistant  
**테스트 환경**: http://10.10.100.45:9097/  
**심각도**: 높음 (High)

---

## 1. 문제 요약

질문 입력 후 기기 선택 UI가 표시될 때, 기기 목록이 많을 경우 스크롤이 되지 않아 하단의 액션 버튼("전체 문서 검색", "선택한 기기로 검색")을 확인하거나 클릭할 수 없는 문제가 발생합니다.

---

## 2. 재현 단계

1. http://10.10.100.45:9097/ 접속
2. 새 대화 시작 또는 기존 대화 열기
3. 질문 입력 (예: "압력 측정 방법")
4. AI 응답 후 기기 선택 UI가 표시됨
5. 기기 목록이 많을 경우 (50개 이상)
6. **문제**: 기기 목록을 스크롤할 수 없고, 하단 버튼이 보이지 않음

---

## 3. 문제 분석

### 3.1 현재 구현 상태

**파일**: `frontend/src/features/chat/components/device-selection-panel.tsx`

```tsx
<Radio.Group
  value={selectedDevice}
  onChange={(e) => setSelectedDevice(e.target.value)}
  style={{ width: "100%" }}
>
  <Space direction="vertical" style={{ width: "100%" }}>
    {devices.map((device) => (
      <Radio key={device.name} value={device.name} ...>
        ...
      </Radio>
    ))}
  </Space>
</Radio.Group>
```

### 3.2 문제점

1. **스크롤 컨테이너 없음**: `Radio.Group`과 `Space` 컴포넌트에 `maxHeight` 및 `overflow-y: auto` 스타일이 없음
2. **고정 높이 제한 없음**: 기기 목록이 화면 높이를 초과해도 스크롤되지 않음
3. **버튼 가림**: 하단 액션 버튼이 기기 목록에 가려져 접근 불가

### 3.3 영향 범위

- **영향 사용자**: 모든 사용자
- **영향 기능**: 기기 선택 기능 전체
- **사용 빈도**: 질문 시 기기 선택이 필요한 경우마다 발생
- **데이터 규모**: 문서에 따르면 기기 종류가 50개 이상 존재

---

## 4. 기대 동작

1. 기기 목록이 많을 경우 스크롤 가능한 영역으로 표시
2. 하단 액션 버튼은 항상 화면에 보이도록 고정
3. 스크롤바가 명확하게 표시되어 사용자가 스크롤 가능함을 인지
4. 선택된 기기가 스크롤 영역 밖에 있어도 자동으로 보이도록 스크롤

---

## 5. 개선 방안

### 5.1 권장 해결책

**방안 A: 기기 목록에 스크롤 컨테이너 추가 (권장)**

```tsx
<Radio.Group
  value={selectedDevice}
  onChange={(e) => setSelectedDevice(e.value)}
  style={{ width: "100%" }}
>
  <div style={{
    maxHeight: "400px",  // 최대 높이 제한
    overflowY: "auto",   // 세로 스크롤 활성화
    overflowX: "hidden",
    paddingRight: "8px", // 스크롤바 공간 확보
  }}>
    <Space direction="vertical" style={{ width: "100%" }}>
      {devices.map((device) => (
        <Radio key={device.name} value={device.name} ...>
          ...
        </Radio>
      ))}
    </Space>
  </div>
</Radio.Group>
```

**장점**:
- 구현이 간단하고 즉시 적용 가능
- 기존 UI 구조 변경 최소화
- 버튼은 항상 보이도록 유지

**단점**:
- 고정 높이로 인한 반응형 제약 가능성

---

**방안 B: 반응형 높이 계산**

```tsx
const [containerHeight, setContainerHeight] = useState(400);

useEffect(() => {
  // 화면 높이에 따라 동적으로 조정
  const calculateHeight = () => {
    const viewportHeight = window.innerHeight;
    const availableHeight = viewportHeight - 400; // 헤더, 질문 영역, 버튼 영역 제외
    setContainerHeight(Math.max(300, Math.min(500, availableHeight)));
  };
  
  calculateHeight();
  window.addEventListener('resize', calculateHeight);
  return () => window.removeEventListener('resize', calculateHeight);
}, []);

<div style={{
  maxHeight: `${containerHeight}px`,
  overflowY: "auto",
  ...
}}>
```

**장점**:
- 다양한 화면 크기에 대응
- 사용자 경험 향상

**단점**:
- 구현 복잡도 증가
- 추가 상태 관리 필요

---

**방안 C: 검색/필터 기능 추가**

기기 목록이 많을 경우 검색 기능을 추가하여 사용자가 원하는 기기를 빠르게 찾을 수 있도록 함.

```tsx
const [searchQuery, setSearchQuery] = useState("");

const filteredDevices = devices.filter(device =>
  device.name.toLowerCase().includes(searchQuery.toLowerCase())
);

// Input 컴포넌트 추가
<Input
  placeholder="기기명으로 검색..."
  value={searchQuery}
  onChange={(e) => setSearchQuery(e.target.value)}
  style={{ marginBottom: 12 }}
  allowClear
/>
```

**장점**:
- 사용자가 원하는 기기를 빠르게 찾을 수 있음
- 스크롤 문제와 함께 사용성 향상

**단점**:
- 추가 개발 시간 필요
- 검색 기능 구현 필요

---

### 5.2 즉시 적용 가능한 해결책 (방안 A)

가장 빠르고 효과적인 해결책은 **방안 A**입니다. 다음 코드 수정을 권장합니다:

**수정 파일**: `frontend/src/features/chat/components/device-selection-panel.tsx`

**수정 위치**: 67-99번째 줄

**수정 내용**:
- `Radio.Group` 내부에 스크롤 가능한 컨테이너 div 추가
- `maxHeight: "400px"` 설정
- `overflowY: "auto"` 설정
- 스크롤바 스타일링 (기존 `styles.css`의 스크롤바 스타일 활용)

---

## 6. 추가 개선 제안

### 6.1 접근성 개선

- 키보드 네비게이션 지원 (Arrow keys로 기기 선택)
- 선택된 기기로 자동 스크롤
- 스크린 리더 지원 (aria-label 추가)

### 6.2 UX 개선

- 기기 개수 표시 (예: "총 50개 기기 중 선택")
- 인기 기기 상단 고정 (문서 수 기준)
- 최근 선택한 기기 하이라이트

### 6.3 성능 최적화

- 가상 스크롤링 (react-window 또는 react-virtualized) 적용 고려
- 기기 목록이 100개 이상일 경우 성능 이슈 가능성

---

## 7. 테스트 체크리스트

수정 후 다음 항목을 테스트해야 합니다:

- [ ] 기기 목록이 10개 이하일 때 정상 표시
- [ ] 기기 목록이 50개 이상일 때 스크롤 가능
- [ ] 하단 버튼이 항상 보임
- [ ] 스크롤바가 명확하게 표시됨
- [ ] 기기 선택 시 정상 동작
- [ ] 모바일 환경에서도 정상 동작
- [ ] 다크 모드에서도 정상 표시
- [ ] 키보드로 스크롤 가능
- [ ] 선택된 기기로 자동 스크롤

---

## 8. 우선순위

**즉시 수정 필요 (P0)**:
- 스크롤 기능 추가 (방안 A)

**단기 개선 (P1)**:
- 반응형 높이 조정 (방안 B)
- 검색 기능 추가 (방안 C)

**중기 개선 (P2)**:
- 접근성 개선
- 성능 최적화

---

## 9. 참고 자료

- 관련 파일:
  - `frontend/src/features/chat/components/device-selection-panel.tsx`
  - `frontend/src/features/chat/pages/chat-page.tsx`
  - `frontend/src/styles.css` (스크롤바 스타일 참고)
- 유사 구현 참고:
  - `frontend/src/styles.css`의 `.review-docs` 클래스 (max-height: 260px, overflow: auto)
  - `frontend/src/features/parsing/components/page-viewer.css`의 스크롤 구현

---

## 10. 결론

기기 선택 UI의 스크롤 문제는 사용자가 기능을 완전히 사용할 수 없게 만드는 심각한 문제입니다. **방안 A**를 즉시 적용하여 스크롤 기능을 추가하는 것을 권장합니다. 이후 사용자 피드백을 바탕으로 추가 개선사항을 적용할 수 있습니다.
