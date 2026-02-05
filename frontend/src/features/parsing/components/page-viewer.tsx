import type { PageData } from "../types";
import "./page-viewer.css";

interface PageViewerProps {
  page: PageData | null;
  isLoading?: boolean;
}

export function PageViewer({ page, isLoading }: PageViewerProps) {
  if (isLoading) {
    return (
      <div className="page-viewer page-viewer--loading">
        <div className="page-viewer__spinner" />
        <span>Loading...</span>
      </div>
    );
  }

  if (!page) {
    return (
      <div className="page-viewer page-viewer--empty">
        <span>Select a page</span>
      </div>
    );
  }

  return (
    <div className="page-viewer">
      <div className="page-viewer__content">
        {/* Left: original image */}
        <div className="page-viewer__panel page-viewer__panel--image">
          <div className="page-viewer__panel-header">
            <h3>Original Image</h3>
            <span className="page-viewer__page-number">Page {page.pageNumber}</span>
          </div>
          <div className="page-viewer__panel-body">
            <img
              src={page.imagePath}
              alt={`Page ${page.pageNumber}`}
              className="page-viewer__image"
            />
          </div>
        </div>

        {/* Right: VLM parse result */}
        <div className="page-viewer__panel page-viewer__panel--vlm">
          <div className="page-viewer__panel-header">
            <h3>VLM Parse Result</h3>
            <span className="page-viewer__char-count">
              {page.vlmText.length.toLocaleString()} chars
            </span>
          </div>
          <div className="page-viewer__panel-body">
            <pre className="page-viewer__vlm-text">{page.vlmText || "(No text)"}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
