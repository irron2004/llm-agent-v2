import { useEffect, useMemo, useState } from "react";
import { Button, Card, Input, Space, Tag, Tabs, Typography } from "antd";

const { Text, Title } = Typography;

type GuidedOption = {
  value: string;
  label: string;
  recommended?: boolean;
};

type GuidedDecision = {
  type: "auto_parse_confirm";
  target_language: "ko" | "en" | "zh" | "ja";
  selected_device?: string | null;
  selected_equip_id?: string | null;
  task_mode: "sop" | "issue" | "all";
};

type GuidedSelectionPanelProps = {
  question: string;
  instruction: string;
  payload: Record<string, unknown>;
  stepIndex?: number;
  draftDecision?: Partial<GuidedDecision>;
  onComplete: (decision: GuidedDecision) => void;
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const asOptions = (value: unknown): GuidedOption[] => {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (!isRecord(item)) return null;
      const v = typeof item.value === "string" ? item.value : "";
      const label = typeof item.label === "string" ? item.label : v;
      if (!v) return null;
      return {
        value: v,
        label,
        recommended: Boolean(item.recommended),
      } satisfies GuidedOption;
    })
    .filter((x): x is GuidedOption => Boolean(x));
};

const normalizeLanguage = (value: string): "ko" | "en" | "zh" | "ja" => {
  const v = value.trim().toLowerCase();
  if (v === "en" || v === "zh" || v === "ja") return v;
  return "ko";
};

const normalizeTaskMode = (value: string): "sop" | "issue" | "all" => {
  const v = value.trim().toLowerCase();
  if (v === "sop" || v === "issue") return v;
  return "all";
};

export function GuidedSelectionPanel({
  question,
  instruction,
  payload,
  stepIndex: externalStepIndex,
  draftDecision,
  onComplete,
}: GuidedSelectionPanelProps) {
  const steps = useMemo(() => {
    const raw = isRecord(payload) ? payload.steps : null;
    if (Array.isArray(raw)) {
      const normalized = raw
        .map((s) => (typeof s === "string" ? s.trim() : ""))
        .filter((s) => s.length > 0);
      if (normalized.length > 0) return normalized;
    }
    return ["language", "device", "equip_id", "task"];
  }, [payload]);

  const options = useMemo(() => {
    const raw = isRecord(payload) ? payload.options : null;
    const opts = isRecord(raw) ? raw : {};
    return {
      language: asOptions(opts.language),
      device: asOptions(opts.device),
      equip_id: asOptions(opts.equip_id),
      task: asOptions(opts.task),
    };
  }, [payload]);

  const defaults = useMemo(() => {
    const raw = isRecord(payload) ? payload.defaults : null;
    const d = isRecord(raw) ? raw : {};
    const target_language = typeof d.target_language === "string" ? d.target_language : "ko";
    const device = typeof d.device === "string" ? d.device : null;
    const equip_id = typeof d.equip_id === "string" ? d.equip_id : null;
    const task_mode = typeof d.task_mode === "string" ? d.task_mode : "all";
    return {
      target_language: normalizeLanguage(target_language),
      device,
      equip_id,
      task_mode: normalizeTaskMode(task_mode),
    };
  }, [payload]);

  const [stepIndex, setStepIndex] = useState(0);
  const [targetLanguage, setTargetLanguage] = useState<GuidedDecision["target_language"]>(defaults.target_language);
  const [selectedDevice, setSelectedDevice] = useState<string | null>(defaults.device);
  const [selectedEquipId, setSelectedEquipId] = useState<string | null>(defaults.equip_id);
  const [taskMode, setTaskMode] = useState<GuidedDecision["task_mode"]>(defaults.task_mode);
  const [manualEquipId, setManualEquipId] = useState("");

  useEffect(() => {
    if (typeof externalStepIndex !== "number") return;
    const clamped = Math.max(0, Math.min(externalStepIndex, steps.length - 1));
    setStepIndex(clamped);
  }, [externalStepIndex, steps.length]);

  useEffect(() => {
    if (!draftDecision) return;
    if (typeof draftDecision.target_language === "string") {
      setTargetLanguage(normalizeLanguage(draftDecision.target_language));
    }
    if ("selected_device" in draftDecision) {
      setSelectedDevice(draftDecision.selected_device ?? null);
    }
    if ("selected_equip_id" in draftDecision) {
      setSelectedEquipId(draftDecision.selected_equip_id ?? null);
      if (typeof draftDecision.selected_equip_id === "string") {
        setManualEquipId(draftDecision.selected_equip_id);
      }
    }
    if (typeof draftDecision.task_mode === "string") {
      setTaskMode(normalizeTaskMode(draftDecision.task_mode));
    }
  }, [draftDecision]);

  const currentStep = steps[Math.min(stepIndex, steps.length - 1)] ?? "language";

  const title =
    currentStep === "language"
      ? "언어 선택"
      : currentStep === "device"
        ? "기기 선택"
        : currentStep === "equip_id"
          ? "설비 ID 선택"
          : "작업 선택";

  const stepOptions: GuidedOption[] =
    currentStep === "language"
      ? options.language
      : currentStep === "device"
        ? options.device
        : currentStep === "equip_id"
          ? options.equip_id
          : options.task;

  const advance = () => {
    setStepIndex((prev) => Math.min(prev + 1, steps.length - 1));
  };

  const applyOption = (opt: GuidedOption) => {
    if (currentStep === "language") {
      setTargetLanguage(normalizeLanguage(opt.value));
      advance();
      return;
    }
    if (currentStep === "device") {
      if (opt.value === "__skip__") {
        setSelectedDevice(null);
      } else {
        setSelectedDevice(opt.value);
      }
      advance();
      return;
    }
    if (currentStep === "equip_id") {
      if (opt.value === "__skip__") {
        setSelectedEquipId(null);
        setManualEquipId("");
        advance();
        return;
      }
      if (opt.value === "__manual__") {
        setSelectedEquipId("__manual__");
        return;
      }
      setSelectedEquipId(opt.value);
      setManualEquipId("");
      advance();
      return;
    }

    const normalizedTask = normalizeTaskMode(opt.value);
    setTaskMode(normalizedTask);
    const equip = selectedEquipId === "__manual__" ? manualEquipId.trim() : selectedEquipId;
    onComplete({
      type: "auto_parse_confirm",
      target_language: targetLanguage,
      selected_device: selectedDevice,
      selected_equip_id: equip && equip.length > 0 ? equip : null,
      task_mode: normalizedTask,
    });
  };

  const hasManualEquip = currentStep === "equip_id" && selectedEquipId === "__manual__";
  const canConfirmManualEquip = manualEquipId.trim().length > 0;

  // 숫자 키 입력으로 옵션 선택
  useEffect(() => {
    if (hasManualEquip) return; // manual input 모드에서는 키보드 비활성

    const handler = (e: KeyboardEvent) => {
      // input/textarea에 포커스 되어있으면 무시
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;

      const num = parseInt(e.key, 10);
      if (isNaN(num)) return;

      if (num === 0) {
        // 0: recommended 항목 선택, 없으면 skip(__skip__) 선택
        const recommended = stepOptions.find((o) => o.recommended);
        const skip = stepOptions.find((o) => o.value === "__skip__");
        const target = recommended ?? skip;
        if (target) applyOption(target);
      } else if (num >= 1 && num <= stepOptions.length) {
        applyOption(stepOptions[num - 1]);
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [stepOptions, hasManualEquip, currentStep, targetLanguage, selectedDevice, selectedEquipId, taskMode, manualEquipId]);

  const optionLabelByValue = useMemo(() => {
    const toMap = (list: GuidedOption[]) => {
      const m = new Map<string, string>();
      for (const item of list) m.set(item.value, item.label);
      return m;
    };
    return {
      language: toMap(options.language),
      device: toMap(options.device),
      equip_id: toMap(options.equip_id),
      task: toMap(options.task),
    };
  }, [options]);

  const selectedLabel = {
    language: optionLabelByValue.language.get(targetLanguage) ?? targetLanguage,
    device: selectedDevice
      ? (optionLabelByValue.device.get(selectedDevice) ?? selectedDevice)
      : "(skip)",
    equip_id: selectedEquipId === "__manual__"
      ? (manualEquipId.trim() || "(manual)")
      : selectedEquipId
        ? (optionLabelByValue.equip_id.get(selectedEquipId) ?? selectedEquipId)
        : "(skip)",
    task: optionLabelByValue.task.get(taskMode) ?? taskMode,
  };

  return (
    <Card
      style={{
        margin: "16px 0 8px",
        borderRadius: 12,
        border: "1px solid var(--color-border)",
        backgroundColor: "var(--color-bg-secondary)",
        maxWidth: 640,
      }}
    >
      <div style={{ marginBottom: 12 }}>
        <Text type="secondary" style={{ fontSize: 13 }}>
          총 {steps.length}개 중 {stepIndex + 1}번째
        </Text>
      </div>
      <Tabs
        activeKey={String(stepIndex)}
        onChange={(key) => setStepIndex(Number(key))}
        items={steps.map((step, idx) => ({
          key: String(idx),
          label:
            step === "language"
              ? "언어"
              : step === "device"
                ? "기기"
                : step === "equip_id"
                  ? "설비"
                  : "작업",
        }))}
        style={{ marginBottom: 16 }}
      />
      <Space direction="vertical" style={{ width: "100%" }} size="middle">
        <div>
          <Title level={5} style={{ margin: 0, marginBottom: 4 }}>{title}</Title>
          <Text type="secondary" style={{ fontSize: 13 }}>{instruction}</Text>
        </div>

        <div
          style={{
            backgroundColor: "var(--color-bg-tertiary)",
            padding: "12px 16px",
            borderRadius: 8,
            fontSize: 13,
          }}
        >
          <Text strong>질문:</Text> {question}
        </div>

        <div>
          <Text type="secondary" style={{ fontSize: 12 }}>
            선택됨: 언어({selectedLabel.language}) / 기기({selectedLabel.device}) / 설비({selectedLabel.equip_id}) / 작업({selectedLabel.task})
          </Text>
        </div>

        <Text type="secondary" style={{ fontSize: 12 }}>
          숫자 선택: 1~{Math.max(stepOptions.length, 1)} (0은 추천 또는 건너뛰기)
        </Text>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {stepOptions.map((opt, index) => {
            const selectedForStep =
              (currentStep === "language" && opt.value === targetLanguage) ||
              (currentStep === "device" && opt.value === (selectedDevice ?? "__skip__")) ||
              (currentStep === "equip_id" && opt.value === (selectedEquipId ?? "__skip__")) ||
              (currentStep === "task" && opt.value === taskMode);

            return (
              <Button
                key={opt.value}
                type={selectedForStep ? "primary" : "default"}
                onClick={() => applyOption(opt)}
              >
                <Space size={6}>
                  <span>{index + 1}. {opt.label}</span>
                  {opt.recommended ? <Tag color="blue">Recommend</Tag> : null}
                  {selectedForStep ? <Tag color="green">Selected</Tag> : null}
                </Space>
              </Button>
            );
          })}
        </div>

        {hasManualEquip ? (
          <Space direction="vertical" style={{ width: "100%" }} size={8}>
            <Input
              placeholder="equip_id를 입력하세요 (예: ABCD-1234)"
              value={manualEquipId}
              onChange={(e) => setManualEquipId(e.target.value)}
            />
            <div style={{ display: "flex", gap: 8 }}>
              <Button
                type="primary"
                disabled={!canConfirmManualEquip}
                onClick={() => {
                  setSelectedEquipId(manualEquipId.trim());
                  advance();
                }}
              >
                확인
              </Button>
              <Button
                onClick={() => {
                  setSelectedEquipId(null);
                  setManualEquipId("");
                  advance();
                }}
              >
                건너뛰기
              </Button>
            </div>
          </Space>
        ) : null}
      </Space>
    </Card>
  );
}
