/**
 * [LEGACY] In-chat equipment/document selection panel
 * Not currently used - replaced by the regeneration panel in the right sidebar
 */
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

  const allDevicesSelected = devices.length > 0 && selectedDevices.length === devices.length;
  const allDocTypesSelected = docTypes.length > 0 && selectedDocTypes.length === docTypes.length;

  // Filter devices by search query
  const filteredDevices = devices.filter((device) =>
    device.name.toLowerCase().includes(searchQuery.toLowerCase())
  );
  const allFilteredSelected = filteredDevices.length > 0 && filteredDevices.every((device) =>
    selectedDevices.includes(device.name)
  );

  const handleToggleDevice = (deviceName: string) => {
    setSelectedDevices((prev) =>
      prev.includes(deviceName)
        ? prev.filter((d) => d !== deviceName)
        : [...prev, deviceName]
    );
  };

  const handleSelectAll = () => {
    const filteredNames = filteredDevices.map((d) => d.name);
    if (allFilteredSelected) {
      setSelectedDevices((prev) => prev.filter((name) => !filteredNames.includes(name)));
    } else {
      setSelectedDevices((prev) => Array.from(new Set([...prev, ...filteredNames])));
    }
  };

  const hasSelection = (
    (devices.length === 0 || selectedDevices.length > 0)
    && (docTypes.length === 0 || selectedDocTypes.length > 0)
  );

  const handleSubmit = () => {
    if (!hasSelection) {
      return;
    }
    onSelect(selectedDevices, selectedDocTypes);
  };

  // Format doc count with comma separators
  const formatDocCount = (count: number) => count.toLocaleString();

  // Calculate total doc count for selected devices
  const selectedDocTypeCount = docTypes
    .filter((d) => selectedDocTypes.includes(d.name))
    .reduce((sum, d) => sum + d.doc_count, 0);

  const selectionLabelParts: string[] = [];
  if (devices.length > 0) {
    if (allDevicesSelected) selectionLabelParts.push("All equipment");
    else if (selectedDevices.length > 0) selectionLabelParts.push("Equipment selected");
  }
  if (docTypes.length > 0) {
    if (allDocTypesSelected) selectionLabelParts.push("All documents");
    else if (selectedDocTypes.length > 0) selectionLabelParts.push(`${selectedDocTypes.length} document types`);
  }
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
            Select Equipment / Document Types
          </Title>
          <Text type="secondary" style={{ fontSize: 13 }}>
            {instruction || "Select equipment and document types to search. Choose at least one of each."}
          </Text>
        </div>

        <div style={{
          backgroundColor: "var(--color-bg-tertiary)",
          padding: "12px 16px",
          borderRadius: 8,
          fontSize: 13,
        }}>
          <Text strong>Question:</Text> {question}
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
                Document types ({docTypes.length})
                {allDocTypesSelected ? (
                  <span style={{ color: "var(--color-accent-primary)", marginLeft: 8 }}>
                    All documents selected
                  </span>
                ) : selectedDocTypes.length > 0 ? (
                  <span style={{ color: "var(--color-accent-primary)", marginLeft: 8 }}>
                    {selectedDocTypes.length} selected ({formatDocCount(selectedDocTypeCount)} documents)
                  </span>
                ) : null}
              </Text>
            </div>
            <div style={{ marginBottom: 8 }}>
              <Button
                size="small"
                type={allDocTypesSelected ? "primary" : "default"}
                onClick={() => {
                  setSelectedDocTypes(allDocTypesSelected ? [] : docTypes.map((d) => d.name));
                }}
              >
                All documents
              </Button>
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
            placeholder="Search by equipment name..."
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
            Equipment selection
            {allDevicesSelected ? (
              <span style={{ color: "var(--color-accent-primary)", marginLeft: 8 }}>
                All equipment selected
              </span>
            ) : selectedDevices.length > 0 ? (
              <span style={{ color: "var(--color-accent-primary)", marginLeft: 8 }}>
                Selected
              </span>
            ) : null}
          </Text>
          <Space size={12}>
            <Button
              type={allDevicesSelected ? "primary" : "default"}
              size="small"
              icon={<GlobalOutlined />}
              onClick={() => {
                setSelectedDevices(allDevicesSelected ? [] : devices.map((d) => d.name));
              }}
            >
              All equipment
            </Button>
            <Button
              type="link"
              size="small"
              onClick={handleSelectAll}
              style={{ padding: 0 }}
            >
              {allFilteredSelected ? "Clear list selection" : "Select all in list"}
            </Button>
          </Space>
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
                      {formatDocCount(device.doc_count)} documents
                    </Text>
                  </div>
                </div>
              );
            })}
          </Space>
        </div>

        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 8 }}>
          <Button
            type="primary"
            onClick={handleSubmit}
            disabled={!hasSelection}
          >
            Search with selection{selectionLabel}
          </Button>
        </div>
      </Space>
    </Card>
  );
}
