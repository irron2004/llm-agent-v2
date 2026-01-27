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
        <span>로딩 중...</span>
      </div>
    );
  }

  if (!page) {
    return (
      <div className="page-viewer page-viewer--empty">
        <span>페이지를 선택해주세요</span>
      </div>
    );
  }

  return (
    <div className="page-viewer">
      <div className="page-viewer__content">
        {/* 왼쪽: 원본 이미지 */}
        <div className="page-viewer__panel page-viewer__panel--image">
          <div className="page-viewer__panel-header">
            <h3>원본 이미지</h3>
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

        {/* 오른쪽: VLM 파싱 결과 */}
        <div className="page-viewer__panel page-viewer__panel--vlm">
          <div className="page-viewer__panel-header">
            <h3>VLM 파싱 결과</h3>
            <span className="page-viewer__char-count">
              {page.vlmText.length.toLocaleString()} 자
            </span>
          </div>
          <div className="page-viewer__panel-body">
            <pre className="page-viewer__vlm-text">{page.vlmText || "(텍스트 없음)"}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
