import { useSearchParams } from "react-router-dom";
import { useIngestionData } from "../hooks/use-ingestion-data";
import { useDocumentList } from "../hooks/use-document-list";
import { PageViewer, PageNavigator } from "../components";
import { Alert, Select } from "antd";
import { env } from "../../../config/env";
import "./parsing-page.css";

export default function ParsingPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // URL 파라미터에서 runId 읽기 (기본값은 env에서)
  const runId = searchParams.get("run") || env.defaultIngestionRun;
  const documentName = searchParams.get("doc") || "";

  // 문서 목록 로드
  const { documents, isLoading: isLoadingDocs } = useDocumentList({ runId });

  // 선택된 문서 (URL에 없으면 첫 번째 문서)
  const selectedDoc = documentName || documents[0] || "";

  const {
    pages,
    sections,
    isLoading,
    error,
    currentPage,
    setCurrentPage,
    totalPages,
  } = useIngestionData({ runId, documentName: selectedDoc });

  const currentPageData = pages[currentPage - 1] || null;

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    const newParams = new URLSearchParams(searchParams);
    newParams.set("page", String(page));
    setSearchParams(newParams, { replace: true });
  };

  const handleDocChange = (doc: string) => {
    const newParams = new URLSearchParams(searchParams);
    newParams.set("doc", doc);
    newParams.delete("page"); // 문서 변경 시 페이지 초기화
    setSearchParams(newParams);
  };

  return (
    <div className="parsing-layout">
      {/* Header */}
      <header className="parsing-header">
        <div className="parsing-header__left">
          <h1 className="parsing-header__title">VLM Parsing Viewer</h1>
          <Select
            className="parsing-header__doc-select"
            value={selectedDoc || undefined}
            onChange={handleDocChange}
            loading={isLoadingDocs}
            placeholder="문서 선택"
            style={{ minWidth: 300 }}
            options={documents.map((doc) => ({ label: doc, value: doc }))}
          />
        </div>
        <div className="parsing-header__right">
          <a href="/" className="parsing-header__link">
            Chat으로 이동
          </a>
        </div>
      </header>

      {/* Navigation */}
      {!isLoading && !error && totalPages > 0 && (
        <PageNavigator
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={handlePageChange}
        />
      )}

      {/* Main Content */}
      <main className="parsing-main">
        {error ? (
          <div className="parsing-error">
            <Alert
              type="error"
              message="데이터 로드 실패"
              description={error}
              showIcon
            />
          </div>
        ) : (
          <PageViewer page={currentPageData} isLoading={isLoading} />
        )}
      </main>

      {/* Footer: 섹션 요약 */}
      {!isLoading && sections.length > 0 && (
        <footer className="parsing-footer">
          <div className="parsing-footer__summary">
            <strong>섹션 수:</strong> {sections.length}개 &nbsp;|&nbsp;
            <strong>총 페이지:</strong> {totalPages}페이지
          </div>
        </footer>
      )}
    </div>
  );
}