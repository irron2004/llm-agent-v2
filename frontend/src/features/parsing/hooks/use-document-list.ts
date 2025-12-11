import { useState, useEffect } from "react";
import { env } from "../../../config/env";

interface UseDocumentListOptions {
  runId: string;
}

interface UseDocumentListReturn {
  documents: string[];
  isLoading: boolean;
  error: string | null;
}

export function useDocumentList({
  runId,
}: UseDocumentListOptions): UseDocumentListReturn {
  const [documents, setDocuments] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) {
      setDocuments([]);
      setIsLoading(false);
      return;
    }

    const loadDocuments = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const indexPath = `${env.ingestionsBase}/${runId}/index.html`;
        const res = await fetch(indexPath);

        if (!res.ok) {
          throw new Error("index.html을 찾을 수 없습니다");
        }

        const html = await res.text();
        const hrefRegex = /href="([^"]+)\/preview\.html"/g;
        const docs: string[] = [];
        let match;

        while ((match = hrefRegex.exec(html)) !== null) {
          const docName = decodeURIComponent(match[1]);
          docs.push(docName);
        }

        setDocuments(docs);
      } catch (err) {
        setError(err instanceof Error ? err.message : "문서 목록 로드 실패");
      } finally {
        setIsLoading(false);
      }
    };

    loadDocuments();
  }, [runId]);

  return { documents, isLoading, error };
}
