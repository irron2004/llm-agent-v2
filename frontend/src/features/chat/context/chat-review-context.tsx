import { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { ReviewDoc, RetrievedDoc } from "../types";

export interface PendingReview {
  threadId: string;
  question: string;
  instruction: string;
  docs: ReviewDoc[];
  searchQueries: string[];
}

interface ChatReviewContextValue {
  pendingReview: PendingReview | null;
  completedRetrievedDocs: RetrievedDoc[] | null;
  selectedRanks: number[];
  editableQueries: string[];
  isEditingQueries: boolean;
  isStreaming: boolean;
  setPendingReview: (review: PendingReview | null) => void;
  setCompletedRetrievedDocs: (docs: RetrievedDoc[] | null) => void;
  setSelectedRanks: React.Dispatch<React.SetStateAction<number[]>>;
  setEditableQueries: React.Dispatch<React.SetStateAction<string[]>>;
  setIsEditingQueries: React.Dispatch<React.SetStateAction<boolean>>;
  setIsStreaming: (streaming: boolean) => void;
  submitReview: (selection: { docIds: string[]; ranks: number[] }) => void;
  submitSearchQueries: (queries: string[]) => void;
  registerSubmitHandlers: (handlers: {
    submitReview: (selection: { docIds: string[]; ranks: number[] }) => void;
    submitSearchQueries: (queries: string[]) => void;
  }) => void;
}

const ChatReviewContext = createContext<ChatReviewContextValue | undefined>(undefined);

export function ChatReviewProvider({ children }: { children: ReactNode }) {
  const [pendingReview, setPendingReviewState] = useState<PendingReview | null>(null);
  const [completedRetrievedDocs, setCompletedRetrievedDocsState] = useState<RetrievedDoc[] | null>(null);
  const [selectedRanks, setSelectedRanks] = useState<number[]>([]);
  const [editableQueries, setEditableQueries] = useState<string[]>([]);
  const [isEditingQueries, setIsEditingQueries] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [submitHandlers, setSubmitHandlers] = useState<{
    submitReview: (selection: { docIds: string[]; ranks: number[] }) => void;
    submitSearchQueries: (queries: string[]) => void;
  } | null>(null);

  const setPendingReview = useCallback((review: PendingReview | null) => {
    setPendingReviewState(review);
    if (review) {
      setSelectedRanks(review.docs.map((doc) => doc.rank));
      setEditableQueries(review.searchQueries);
      setIsEditingQueries(false);
      // Clear completed docs when pending review appears
      setCompletedRetrievedDocsState(null);
    } else {
      setSelectedRanks([]);
      setEditableQueries([]);
      setIsEditingQueries(false);
    }
  }, []);

  const setCompletedRetrievedDocs = useCallback((docs: RetrievedDoc[] | null) => {
    setCompletedRetrievedDocsState(docs);
    // Clear pending review when completed docs are set
    if (docs) {
      setPendingReviewState(null);
    }
  }, []);

  const registerSubmitHandlers = useCallback((handlers: {
    submitReview: (selection: { docIds: string[]; ranks: number[] }) => void;
    submitSearchQueries: (queries: string[]) => void;
  }) => {
    setSubmitHandlers(handlers);
  }, []);

  const submitReview = useCallback((selection: { docIds: string[]; ranks: number[] }) => {
    submitHandlers?.submitReview(selection);
  }, [submitHandlers]);

  const submitSearchQueries = useCallback((queries: string[]) => {
    submitHandlers?.submitSearchQueries(queries);
  }, [submitHandlers]);

  return (
    <ChatReviewContext.Provider
      value={{
        pendingReview,
        completedRetrievedDocs,
        selectedRanks,
        editableQueries,
        isEditingQueries,
        isStreaming,
        setPendingReview,
        setCompletedRetrievedDocs,
        setSelectedRanks,
        setEditableQueries,
        setIsEditingQueries,
        setIsStreaming,
        submitReview,
        submitSearchQueries,
        registerSubmitHandlers,
      }}
    >
      {children}
    </ChatReviewContext.Provider>
  );
}

export function useChatReview() {
  const context = useContext(ChatReviewContext);
  if (context === undefined) {
    throw new Error("useChatReview must be used within ChatReviewProvider");
  }
  return context;
}
