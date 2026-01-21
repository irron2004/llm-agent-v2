import { useState, ChangeEvent } from "react";
import { Button, Card, Checkbox, Input, Space, Typography } from "antd";
import { LaptopOutlined, GlobalOutlined, SearchOutlined } from "@ant-design/icons";

const { Text, Title } = Typography;

type DeviceInfo = {
  name: string;
  doc_count: number;
};

type DocTypeInfo = {
  name: string;
  doc_count: number;
};

type DeviceSelectionPanelProps = {
  question: string;
  devices: DeviceInfo[];
  docTypes?: DocTypeInfo[];
  onSelect: (devices: string[], docTypes: string[]) => void;
  instruction?: string;
};

export function DeviceSelectionPanel({
  question,
  devices,
  docTypes = [],
  onSelect,
  instruction,
}: DeviceSelectionPanelProps) {
  const [selectedDevices, setSelectedDevices] = useState<string[]>([]);
  const [selectedDocTypes, setSelectedDocTypes] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState("");

  // Filter devices by search query
  const filteredDevices = devices.filter((device) =>
    device.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleToggleDevice = (deviceName: string) => {
    setSelectedDevices((prev) =>
      prev.includes(deviceName)
        ? prev.filter((d) => d !== deviceName)
        : [...prev, deviceName]
    );
  };

  const handleSelectAll = () => {
    if (selectedDevices.length === filteredDevices.length) {
      setSelectedDevices([]);
    } else {
      setSelectedDevices(filteredDevices.map((d) => d.name));
    }
  };

  const handleSubmit = () => {
    onSelect(selectedDevices, selectedDocTypes);
  };

  const handleSkip = () => {
    onSelect([], selectedDocTypes);
  };

  // Format doc count with comma separators
  const formatDocCount = (count: number) => count.toLocaleString();

  // Calculate total doc count for selected devices
  const selectedDocCount = devices
    .filter((d) => selectedDevices.includes(d.name))
    .reduce((sum, d) => sum + d.doc_count, 0);
  const selectedDocTypeCount = docTypes
    .filter((d) => selectedDocTypes.includes(d.name))
    .reduce((sum, d) => sum + d.doc_count, 0);

  const hasSelection = selectedDevices.length > 0 || selectedDocTypes.length > 0;
  const selectionLabelParts: string[] = [];
  if (selectedDevices.length > 0) selectionLabelParts.push(`기기 ${selectedDevices.length}개`);
  if (selectedDocTypes.length > 0) selectionLabelParts.push(`문서 ${selectedDocTypes.length}종`);
  const selectionLabel = selectionLabelParts.length > 0
    ? ` (${selectionLabelParts.join(", ")})`
    : "";

  const handleToggleDocType = (docTypeName: string) => {
    setSelectedDocTypes((prev) =>
      prev.includes(docTypeName)
        ? prev.filter((d) => d !== docTypeName)
        : [...prev, docTypeName]
    );
  };

  return (
    <Card
      style={{
        margin: "16px 0",
        borderRadius: 12,
        border: "1px solid var(--color-border)",
        backgroundColor: "var(--color-bg-secondary)",
      }}
    >
      <Space direction="vertical" style={{ width: "100%" }} size="middle">
        <div>
          <Title level={5} style={{ margin: 0, marginBottom: 4 }}>
            <LaptopOutlined style={{ marginRight: 8 }} />
            기기/문서 종류 선택
          </Title>
          <Text type="secondary" style={{ fontSize: 13 }}>
            {instruction || "검색할 기기/문서 종류를 선택하세요. 선택한 기기 문서 10개 + 전체 문서 10개를 검색합니다."}
          </Text>
        </div>

        <div style={{
          backgroundColor: "var(--color-bg-tertiary)",
          padding: "12px 16px",
          borderRadius: 8,
          fontSize: 13,
        }}>
          <Text strong>질문:</Text> {question}
        </div>

        {docTypes.length > 0 && (
          <div
            style={{
              backgroundColor: "var(--color-bg-tertiary)",
              padding: "10px 12px",
              borderRadius: 8,
              border: "1px solid var(--color-border)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                문서 종류 {docTypes.length}종
                {selectedDocTypes.length > 0 && (
                  <span style={{ color: "var(--color-accent-primary)", marginLeft: 8 }}>
                    {selectedDocTypes.length}종 선택 ({formatDocCount(selectedDocTypeCount)} 문서)
                  </span>
                )}
              </Text>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {docTypes.map((docType) => {
                const isSelected = selectedDocTypes.includes(docType.name);
                return (
                  <div
                    key={docType.name}
                    onClick={() => handleToggleDocType(docType.name)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      padding: "6px 10px",
                      borderRadius: 999,
                      border: isSelected
                        ? "1px solid var(--color-accent-primary)"
                        : "1px solid var(--color-border)",
                      backgroundColor: isSelected
                        ? "var(--color-accent-primary-light)"
                        : "var(--color-bg-primary)",
                      cursor: "pointer",
                      userSelect: "none",
                    }}
                  >
                    <Checkbox checked={isSelected} />
                    <span style={{ fontSize: 12, fontWeight: 500 }}>{docType.name}</span>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {formatDocCount(docType.doc_count)}
                    </Text>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Search input - shown when there are many devices */}
        {devices.length > 10 && (
          <Input
            placeholder="기기명으로 검색..."
            prefix={<SearchOutlined style={{ color: "var(--color-text-secondary)" }} />}
            value={searchQuery}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
            allowClear
            style={{ marginBottom: 4 }}
          />
        )}

        {/* Device count and select all */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            총 {devices.length}개 기기
            {searchQuery && ` (${filteredDevices.length}개 일치)`}
            {selectedDevices.length > 0 && (
              <span style={{ color: "var(--color-accent-primary)", marginLeft: 8 }}>
                {selectedDevices.length}개 선택 ({formatDocCount(selectedDocCount)} 문서)
              </span>
            )}
          </Text>
          <Button
            type="link"
            size="small"
            onClick={handleSelectAll}
            style={{ padding: 0 }}
          >
            {selectedDevices.length === filteredDevices.length ? "전체 해제" : "전체 선택"}
          </Button>
        </div>

        {/* Scrollable device list with checkboxes */}
        <div
          style={{
            maxHeight: "400px",
            overflowY: "auto",
            overflowX: "hidden",
            paddingRight: "4px",
          }}
        >
          <Space direction="vertical" style={{ width: "100%" }}>
            {filteredDevices.map((device) => {
              const isSelected = selectedDevices.includes(device.name);
              return (
                <div
                  key={device.name}
                  onClick={() => handleToggleDevice(device.name)}
                  style={{
                    width: "100%",
                    padding: "10px 12px",
                    borderRadius: 8,
                    border: isSelected
                      ? "2px solid var(--color-accent-primary)"
                      : "1px solid var(--color-border)",
                    backgroundColor: isSelected
                      ? "var(--color-accent-primary-light)"
                      : "var(--color-bg-primary)",
                    transition: "all 0.2s ease",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                  }}
                >
                  <Checkbox checked={isSelected} />
                  <div style={{ flex: 1, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontWeight: 500 }}>{device.name}</span>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {formatDocCount(device.doc_count)} 문서
                    </Text>
                  </div>
                </div>
              );
            })}
          </Space>
        </div>

        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 8 }}>
          <Button
            icon={<GlobalOutlined />}
            onClick={handleSkip}
          >
            기기 선택 건너뛰기
          </Button>
          <Button
            type="primary"
            onClick={handleSubmit}
            disabled={!hasSelection}
          >
            선택한 조건으로 검색{selectionLabel}
          </Button>
        </div>
      </Space>
    </Card>
  );
}
