import { useState, useEffect } from "react";
import { buildUrl } from "../../../config/env";

interface RunFolder {
  name: string;
  path: string;
}

interface UseRunFoldersReturn {
  folders: RunFolder[];
  isLoading: boolean;
  error: string | null;
}

export function useRunFolders(): UseRunFoldersReturn {
  const [folders, setFolders] = useState<RunFolder[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadFolders = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const res = await fetch(buildUrl("/api/ingestions/runs"));

        if (!res.ok) {
          throw new Error(`Failed to load run folders: ${res.status}`);
        }

        const data = await res.json();
        setFolders(data.folders || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "폴더 목록 로드 실패");
      } finally {
        setIsLoading(false);
      }
    };

    loadFolders();
  }, []);

  return { folders, isLoading, error };
}
