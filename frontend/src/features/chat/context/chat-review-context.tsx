import { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { ReviewDoc, RetrievedDoc } from "../types";

export interface PendingReview {
  threadId: string;
  question: string;
  instruction: string;
  docs: ReviewDoc[];
  searchQueries: string[];
}

export interface PendingRegeneration {
  messageId: string;
  originalQuery: string;
  docs: RetrievedDoc[];
  searchQueries: string[];
  selectedDevices: string[];
  selectedDocTypes: string[];
}

interface ChatReviewContextValue {
  pendingReview: PendingReview | null;
  pendingRegeneration: PendingRegeneration | null;
  completedRetrievedDocs: RetrievedDoc[] | null;
  selectedRanks: number[];
  editableQueries: string[];
  isEditingQueries: boolean;
  isStreaming: boolean;
  setPendingReview: (review: PendingReview | null) => void;
  setPendingRegeneration: (regen: PendingRegeneration | null) => void;
  setCompletedRetrievedDocs: (docs: RetrievedDoc[] | null) => void;
  setSelectedRanks: React.Dispatch<React.SetStateAction<number[]>>;
  setEditableQueries: React.Dispatch<React.SetStateAction<string[]>>;
  setIsEditingQueries: React.Dispatch<React.SetStateAction<boolean>>;
  setIsStreaming: (streaming: boolean) => void;
  submitReview: (selection: { docIds: string[]; ranks: number[] }) => void;
  submitSearchQueries: (queries: string[]) => void;
  submitRegeneration: (payload: {
    originalQuery: string;
    searchQueries: string[];
    selectedDevices: string[];
    selectedDocTypes: string[];
    selectedDocIds: string[];
  }) => void;
  registerSubmitHandlers: (handlers: {
    submitReview: (selection: { docIds: string[]; ranks: number[] }) => void;
    submitSearchQueries: (queries: string[]) => void;
  }) => void;
  registerRegenerationHandlers: (handlers: {
    submitRegeneration: (payload: {
      originalQuery: string;
      searchQueries: string[];
      selectedDevices: string[];
      selectedDocTypes: string[];
      selectedDocIds: string[];
    }) => void;
  }) => void;
}

const ChatReviewContext = createContext<ChatReviewContextValue | undefined>(undefined);

export function ChatReviewProvider({ children }: { children: ReactNode }) {
  const [pendingReview, setPendingReviewState] = useState<PendingReview | null>(null);
  const [pendingRegeneration, setPendingRegenerationState] = useState<PendingRegeneration | null>(null);
  const [completedRetrievedDocs, setCompletedRetrievedDocsState] = useState<RetrievedDoc[] | null>(null);
  const [selectedRanks, setSelectedRanks] = useState<number[]>([]);
  const [editableQueries, setEditableQueries] = useState<string[]>([]);
  const [isEditingQueries, setIsEditingQueries] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [submitHandlers, setSubmitHandlers] = useState<{
    submitReview: (selection: { docIds: string[]; ranks: number[] }) => void;
    submitSearchQueries: (queries: string[]) => void;
  } | null>(null);
  const [regenerationHandlers, setRegenerationHandlers] = useState<{
    submitRegeneration: (payload: {
      originalQuery: string;
      searchQueries: string[];
      selectedDevices: string[];
      selectedDocTypes: string[];
      selectedDocIds: string[];
    }) => void;
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

  const setPendingRegeneration = useCallback((regen: PendingRegeneration | null) => {
    setPendingRegenerationState(regen);
    if (regen) {
      // Reset review-related selections when regeneration panel opens
      setPendingReviewState(null);
      setCompletedRetrievedDocsState(null);
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

  const registerRegenerationHandlers = useCallback((handlers: {
    submitRegeneration: (payload: {
      originalQuery: string;
      searchQueries: string[];
      selectedDevices: string[];
      selectedDocTypes: string[];
      selectedDocIds: string[];
    }) => void;
  }) => {
    setRegenerationHandlers(handlers);
  }, []);

  const submitReview = useCallback((selection: { docIds: string[]; ranks: number[] }) => {
    submitHandlers?.submitReview(selection);
  }, [submitHandlers]);

  const submitSearchQueries = useCallback((queries: string[]) => {
    submitHandlers?.submitSearchQueries(queries);
  }, [submitHandlers]);

  const submitRegeneration = useCallback((payload: {
    originalQuery: string;
    searchQueries: string[];
    selectedDevices: string[];
    selectedDocTypes: string[];
    selectedDocIds: string[];
  }) => {
    regenerationHandlers?.submitRegeneration(payload);
  }, [regenerationHandlers]);

  return (
    <ChatReviewContext.Provider
      value={{
        pendingReview,
        pendingRegeneration,
        completedRetrievedDocs,
        selectedRanks,
        editableQueries,
        isEditingQueries,
        isStreaming,
        setPendingReview,
        setPendingRegeneration,
        setCompletedRetrievedDocs,
        setSelectedRanks,
        setEditableQueries,
        setIsEditingQueries,
        setIsStreaming,
        submitReview,
        submitSearchQueries,
        submitRegeneration,
        registerSubmitHandlers,
        registerRegenerationHandlers,
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
