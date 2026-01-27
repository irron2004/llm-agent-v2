export interface IngestionRun {
  id: string; // timestamp folder name (e.g., "20251211_070055")
  path: string;
  documents: IngestionDocument[];
}

export interface IngestionDocument {
  name: string;
  path: string;
  pageCount: number;
}

export interface PageData {
  pageNumber: number;
  imagePath: string;
  vlmText: string;
}

export interface Section {
  title: string;
  text: string;
  page_start: number | null;
  page_end: number | null;
  metadata?: Record<string, unknown>;
}