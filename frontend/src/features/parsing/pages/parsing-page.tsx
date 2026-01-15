import { useSearchParams } from "react-router-dom";
import { useIngestionData } from "../hooks/use-ingestion-data";
import { useDocumentList } from "../hooks/use-document-list";
import { useRunFolders } from "../hooks/use-run-folders";
import { PageViewer, PageNavigator } from "../components";
import { Alert, Select, Space } from "antd";
import { env } from "../../../config/env";
import "./parsing-page.css";

export default function ParsingPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Run 폴더 목록 가져오기
  const { folders, isLoading: isLoadingFolders } = useRunFolders();

  const runId = searchParams.get("run") || env.defaultIngestionRun || folders[0]?.name || "";
  const documentName = searchParams.get("doc") || "";

  const { documents, isLoading: isLoadingDocs } = useDocumentList({ runId });
  const selectedDoc = documentName || documents[0] || "";

  const handleRunChange = (run: string) => {
    const newParams = new URLSearchParams();
    newParams.set("run", run);
    setSearchParams(newParams);
  };

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
    newParams.delete("page");
    setSearchParams(newParams);
  };

  return (
    <div className="parsing-layout">
      <header className="parsing-header">
        <div className="parsing-header__left">
          <h1 className="parsing-header__title">VLM Parsing Viewer</h1>
          <Space size="middle">
            <Select
              className="parsing-header__run-select"
              value={runId || undefined}
              onChange={handleRunChange}
              loading={isLoadingFolders}
              placeholder="Run 폴더 선택"
              style={{ minWidth: 200 }}
              options={folders.map((f) => ({ label: f.name, value: f.name }))}
            />
            <Select
              className="parsing-header__doc-select"
              value={selectedDoc || undefined}
              onChange={handleDocChange}
              loading={isLoadingDocs}
              placeholder="문서 선택"
              style={{ minWidth: 300 }}
              options={documents.map((doc) => ({ label: doc, value: doc }))}
              disabled={!runId}
            />
          </Space>
        </div>
        <div className="parsing-header__right">
          <a href="/" className="parsing-header__link">
            Chat으로 이동
          </a>
        </div>
      </header>

      {!isLoading && !error && totalPages > 0 && (
        <PageNavigator
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={handlePageChange}
        />
      )}

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
