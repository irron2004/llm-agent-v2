import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { SendOutlined, StopOutlined } from "@ant-design/icons";
import { Select } from "antd";

type SelectOption = { label: string; value: string };
type DocTypeOption = SelectOption & { isPreset?: boolean };

type ChatInputProps = {
  onSend: (message: string) => void;
  onStop?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
  placeholder?: string;
  docTypeOptions?: DocTypeOption[];
  selectedDocTypes?: string[];
  onDocTypesChange?: (types: string[]) => void;
  modelOptions?: SelectOption[];
  selectedModel?: string | null;
  onModelChange?: (model: string | null) => void;
  equipOptions?: SelectOption[];
  selectedEquip?: string | null;
  onEquipChange?: (equip: string | null) => void;
};

export function ChatInput({
  onSend,
  onStop,
  isStreaming = false,
  disabled = false,
  placeholder = "메시지를 입력하세요...",
  docTypeOptions,
  selectedDocTypes,
  onDocTypesChange,
  modelOptions,
  selectedModel,
  onModelChange,
  equipOptions,
  selectedEquip,
  onEquipChange,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [value]);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled || isStreaming) return;
    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const hasFilterBar = Boolean(docTypeOptions || modelOptions || equipOptions);

  return (
    <div className="input-wrapper">
      {hasFilterBar && (
        <div style={{ display: "flex", gap: 8, padding: "8px 0", flexWrap: "wrap", alignItems: "center" }}>
          {docTypeOptions && onDocTypesChange && (
            <Select
              mode="multiple"
              allowClear
              placeholder="Doc"
              style={{ minWidth: 160, flex: "1 1 160px" }}
              options={docTypeOptions}
              value={selectedDocTypes ?? []}
              onChange={onDocTypesChange}
              maxTagCount="responsive"
              optionRender={(option) => (
                <span style={{ fontWeight: (option.data as DocTypeOption).isPreset ? 700 : 400 }}>
                  {option.label}
                </span>
              )}
            />
          )}
          {modelOptions && onModelChange && (
            <Select
              showSearch
              allowClear
              placeholder="Model"
              style={{ minWidth: 160, flex: "1 1 160px" }}
              options={modelOptions}
              value={selectedModel ?? undefined}
              onChange={(v) => onModelChange(v ?? null)}
              filterOption={(input, option) =>
                (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
              }
            />
          )}
          {equipOptions && onEquipChange && (
            <Select
              showSearch
              allowClear
              placeholder="Equip"
              style={{ minWidth: 140, flex: "1 1 140px" }}
              options={equipOptions}
              value={selectedEquip ?? undefined}
              onChange={(v) => onEquipChange(v ?? null)}
              filterOption={(input, option) =>
                (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
              }
            />
          )}
        </div>
      )}
      <textarea
        ref={textareaRef}
        className="input-textarea"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
      />
      {isStreaming ? (
        <button className="send-button" onClick={onStop} style={{ backgroundColor: "var(--color-error)" }}>
          <StopOutlined />
          <span>Stop</span>
        </button>
      ) : (
        <button className="send-button" onClick={handleSend} disabled={!value.trim() || disabled}>
          <SendOutlined />
          <span>Send</span>
        </button>
      )}
    </div>
  );
}
