import { SuggestedDevice } from "../types";

type DeviceSuggestionsProps = {
  devices: SuggestedDevice[];
  onSelect?: (index: number, deviceName: string) => void;
};

export function DeviceSuggestions({ devices, onSelect }: DeviceSuggestionsProps) {
  if (!devices || devices.length === 0) return null;

  return (
    <div className="device-suggestions">
      <p className="device-suggestions-title">
        검색된 문서의 장비 목록 (번호 입력으로 선택):
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
  );
}
