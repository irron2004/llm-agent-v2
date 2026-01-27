import { useState, useEffect, useCallback } from "react";
import type { PageData, Section } from "../types";
import { env } from "../../../config/env";

interface UseIngestionDataOptions {
  runId: string;
  documentName: string;
}

interface UseIngestionDataReturn {
  pages: PageData[];
  sections: Section[];
  isLoading: boolean;
  error: string | null;
  currentPage: number;
  setCurrentPage: (page: number) => void;
  totalPages: number;
}

export function useIngestionData({
  runId,
  documentName,
}: UseIngestionDataOptions): UseIngestionDataReturn {
  const [pages, setPages] = useState<PageData[]>([]);
  const [sections, setSections] = useState<Section[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);

  const basePath = `${env.ingestionsBase}/${runId}/${documentName}`;

  // 페이지 데이터 로드
  const loadPageData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // sections.json에서 페이지 수 파악
      const sectionsRes = await fetch(`${basePath}/sections.json`);
      if (!sectionsRes.ok) {
        throw new Error("sections.json을 찾을 수 없습니다");
      }
      const sectionsData: Section[] = await sectionsRes.json();
      setSections(sectionsData);

      // 페이지 파일들 탐색 (최대 100페이지까지 시도)
      const pageDataList: PageData[] = [];
      for (let i = 1; i <= 100; i++) {
        const pageNum = String(i).padStart(3, "0");
        const imagePath = `${basePath}/pages/page_${pageNum}.png`;
        const vlmPath = `${basePath}/vlm/page_${pageNum}.txt`;

        // 이미지 존재 확인
        const imgRes = await fetch(imagePath, { method: "HEAD" });
        if (!imgRes.ok) break;

        // VLM 텍스트 로드
        let vlmText = "";
        try {
          const vlmRes = await fetch(vlmPath);
          if (vlmRes.ok) {
            vlmText = await vlmRes.text();
          }
        } catch {
          // VLM 텍스트가 없어도 계속 진행
        }

        pageDataList.push({
          pageNumber: i,
          imagePath,
          vlmText,
        });
      }

      setPages(pageDataList);
    } catch (err) {
      setError(err instanceof Error ? err.message : "데이터 로드 실패");
    } finally {
      setIsLoading(false);
    }
  }, [basePath]);

  useEffect(() => {
    if (runId && documentName) {
      loadPageData();
    }
  }, [runId, documentName, loadPageData]);

  return {
    pages,
    sections,
    isLoading,
    error,
    currentPage,
    setCurrentPage,
    totalPages: pages.length,
  };
}
