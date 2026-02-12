import { useEffect, useState, useRef } from "react";
import { SuggestedDevice } from "../types";

type DeviceSuggestionsProps = {
  devices: SuggestedDevice[];
  onSelect?: (index: number, deviceName: string) => void;
  onDismiss?: () => void;  // ESC로 닫기
  isActive?: boolean;  // 활성화 상태 (마지막 메시지에서만)
};

type Step = "confirm" | "select";

export function DeviceSuggestions({ devices, onSelect, onDismiss, isActive = false }: DeviceSuggestionsProps) {
  // 2단계 플로우: confirm -> select
  const [step, setStep] = useState<Step>("confirm");
  const listRef = useRef<HTMLDivElement>(null);

  // isActive가 변경되면 step을 초기화
  useEffect(() => {
    if (isActive) {
      setStep("confirm");
    }
  }, [isActive]);

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

      if (step === "confirm") {
        // confirm 단계: 1 = Yes (기기 선택으로 이동), 2 = No (닫기)
        if (e.key === "1") {
          e.preventDefault();
          e.stopPropagation();
          setStep("select");
        } else if (e.key === "2") {
          e.preventDefault();
          e.stopPropagation();
          onDismiss?.();
        }
      } else if (step === "select") {
        // select 단계: 숫자 키 1-9로 기기 선택
        const key = e.key;
        if (key >= "1" && key <= "9" && onSelect) {
          const index = parseInt(key, 10);
          if (index <= devices.length) {
            e.preventDefault();
            e.stopPropagation();
            onSelect(index, devices[index - 1].name);
          }
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [isActive, devices, onSelect, onDismiss, step]);

  if (!devices || devices.length === 0) return null;

  // Show compact view when inactive
  if (!isActive) {
    return (
      <div className="device-suggestions device-suggestions-inactive">
        <p className="device-suggestions-title-small">
          Detected equipment: {devices.map(d => d.name).join(", ")}
        </p>
      </div>
    );
  }

  // Step 1: 확인 단계 - 기기 필터 검색 여부 물어보기
  if (step === "confirm") {
    return (
      <div className="device-suggestions-overlay">
        <div className="device-suggestions-modal device-suggestions-confirm">
          <p className="device-suggestions-title">
            Should I search for a specific equipment?
          </p>
          <p className="device-suggestions-hint">
            Press <kbd>ESC</kbd> to go back to the chat
          </p>
          <div className="device-suggestions-confirm-buttons">
            <button
              className="device-confirm-button device-confirm-yes"
              onClick={() => setStep("select")}
            >
              <span className="device-number">1.</span>
              <span>Yes</span>
            </button>
            <button
              className="device-confirm-button device-confirm-no"
              onClick={() => onDismiss?.()}
            >
              <span className="device-number">2.</span>
              <span>No</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Step 2: select equipment
  return (
    <div className="device-suggestions-overlay">
      <div className="device-suggestions-modal">
        <p className="device-suggestions-title">
          Which equipment would you like to search for?
        </p>
        <p className="device-suggestions-hint">
          Press <kbd>ESC</kbd> if you don't want additional search.
        </p>
        <div
          ref={listRef}
          className="device-suggestions-list"
          tabIndex={0}
          style={{
            height: '280px',
            maxHeight: '280px',
            overflowY: 'scroll',
            overflowX: 'hidden',
          }}
        >
          {devices.map((device, index) => (
            <button
              key={device.name}
              className="device-suggestion-item"
              onClick={() => onSelect?.(index + 1, device.name)}
            >
              <span className="device-number">{index + 1}.</span>
              <span className="device-name">{device.name}</span>
              <span className="device-count">({device.count} results)</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
