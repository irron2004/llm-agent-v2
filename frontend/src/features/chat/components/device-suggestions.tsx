import { useEffect } from "react";
import { SuggestedDevice } from "../types";

type DeviceSuggestionsProps = {
  devices: SuggestedDevice[];
  onSelect?: (index: number, deviceName: string) => void;
  onDismiss?: () => void;  // ESC로 닫기
  isActive?: boolean;  // 활성화 상태 (마지막 메시지에서만)
};

export function DeviceSuggestions({ devices, onSelect, onDismiss, isActive = false }: DeviceSuggestionsProps) {
  // 숫자 키 및 ESC 키 처리 (isActive일 때만)
  useEffect(() => {
    if (!isActive || !devices || devices.length === 0) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // ESC 키: 닫기
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        onDismiss?.();
        return;
      }

      // 숫자 키 1-9: 기기 선택
      const key = e.key;
      if (key >= "1" && key <= "9" && onSelect) {
        const index = parseInt(key, 10);
        if (index <= devices.length) {
          e.preventDefault();
          e.stopPropagation();
          onSelect(index, devices[index - 1].name);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [isActive, devices, onSelect, onDismiss]);

  if (!devices || devices.length === 0) return null;

  // 활성화 상태가 아니면 작은 버전으로 표시
  if (!isActive) {
    return (
      <div className="device-suggestions device-suggestions-inactive">
        <p className="device-suggestions-title-small">
          검색된 기기: {devices.map(d => d.name).join(", ")}
        </p>
      </div>
    );
  }

  // 활성화 상태: 크게 표시
  return (
    <div className="device-suggestions-overlay">
      <div className="device-suggestions-modal">
        <p className="device-suggestions-title">
          다음의 기기 중에서 어떤 기기를 중심으로 다시 검색할까요?
        </p>
        <p className="device-suggestions-hint">
          추가 검색을 원하지 않으면 <kbd>ESC</kbd>를 눌러주세요.
        </p>
        <div className="device-suggestions-list">
          {devices.map((device, index) => (
            <button
              key={device.name}
              className="device-suggestion-item"
              onClick={() => onSelect?.(index + 1, device.name)}
            >
              <span className="device-number">{index + 1}.</span>
              <span className="device-name">{device.name}</span>
              <span className="device-count">({device.count}건)</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
