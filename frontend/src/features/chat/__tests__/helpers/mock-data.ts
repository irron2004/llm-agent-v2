import type { DeviceCatalogResponse } from "../../types";
import type { PendingRegeneration } from "../../context/chat-review-context";

export function makePendingRegeneration(
  overrides: Partial<PendingRegeneration> = {},
): PendingRegeneration {
  return {
    messageId: "msg-1",
    originalQuery: "pump vibration 원인",
    docs: [],
    searchQueries: ["pump vibration"],
    selectedDevices: [],
    selectedDocTypes: [],
    reason: "missing_device_parse",
    ...overrides,
  };
}

export function makeDeviceCatalogResponse(
  overrides: Partial<DeviceCatalogResponse> = {},
): DeviceCatalogResponse {
  return {
    devices: [
      { name: "Pump-A100", doc_count: 120 },
      { name: "Compressor-B200", doc_count: 85 },
      { name: "Turbine-C300", doc_count: 200 },
    ],
    doc_types: [
      { name: "Manual", doc_count: 300 },
      { name: "P&ID", doc_count: 150 },
    ],
    ...overrides,
  };
}
