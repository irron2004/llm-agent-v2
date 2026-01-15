import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  SearchOutlined,
  MessageOutlined,
  ExperimentOutlined,
  FileTextOutlined,
} from "@ant-design/icons";
import { useGlobalSearch } from "./global-search-context";
import { useChatHistoryContext } from "../../features/chat/context/chat-history-context";
import "./global-search.css";

interface SearchItem {
  id: string;
  type: "page" | "chat";
  title: string;
  subtitle?: string;
  icon: React.ReactNode;
  path?: string;
}

// Static page routes for navigation
const PAGE_ITEMS: SearchItem[] = [
  {
    id: "page-chat",
    type: "page",
    title: "Chat",
    subtitle: "Start a new conversation",
    icon: <MessageOutlined />,
    path: "/",
  },
  {
    id: "page-search",
    type: "page",
    title: "Search",
    subtitle: "Search documents",
    icon: <SearchOutlined />,
    path: "/search",
  },
  {
    id: "page-retrieval",
    type: "page",
    title: "Retrieval Test",
    subtitle: "Test retrieval performance",
    icon: <ExperimentOutlined />,
    path: "/retrieval-test",
  },
  {
    id: "page-parsing",
    type: "page",
    title: "Parsing",
    subtitle: "Document parsing",
    icon: <FileTextOutlined />,
    path: "/parsing",
  },
];

export function GlobalSearch() {
  const { isOpen, close } = useGlobalSearch();
  const navigate = useNavigate();
  const { history } = useChatHistoryContext();
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Convert chat history to search items
  const chatItems: SearchItem[] = useMemo(() => {
    return history.map((chat) => ({
      id: `chat-${chat.id}`,
      type: "chat" as const,
      title: chat.title,
      subtitle: formatRelativeTime(chat.updatedAt),
      icon: <MessageOutlined />,
      path: "/", // TODO: Navigate to specific chat
    }));
  }, [history]);

  // Filter items based on query
  const filteredItems = useMemo(() => {
    const normalizedQuery = query.toLowerCase().trim();

    if (!normalizedQuery) {
      return {
        pages: PAGE_ITEMS,
        chats: chatItems.slice(0, 5), // Show recent 5 chats
      };
    }

    return {
      pages: PAGE_ITEMS.filter(
        (item) =>
          item.title.toLowerCase().includes(normalizedQuery) ||
          item.subtitle?.toLowerCase().includes(normalizedQuery)
      ),
      chats: chatItems.filter((item) =>
        item.title.toLowerCase().includes(normalizedQuery)
      ),
    };
  }, [query, chatItems]);

  // Flatten items for keyboard navigation
  const allItems = useMemo(() => {
    return [...filteredItems.pages, ...filteredItems.chats];
  }, [filteredItems]);

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      // Focus input after modal animation
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Keep selected index in bounds
  useEffect(() => {
    if (selectedIndex >= allItems.length) {
      setSelectedIndex(Math.max(0, allItems.length - 1));
    }
  }, [allItems.length, selectedIndex]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) => Math.min(prev + 1, allItems.length - 1));
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) => Math.max(prev - 1, 0));
          break;
        case "Enter":
          e.preventDefault();
          if (allItems[selectedIndex]) {
            handleItemSelect(allItems[selectedIndex]);
          }
          break;
      }
    },
    [allItems, selectedIndex]
  );

  // Handle item selection
  const handleItemSelect = useCallback(
    (item: SearchItem) => {
      if (item.path) {
        navigate(item.path);
      }
      close();
    },
    [navigate, close]
  );

  // Handle overlay click (close on backdrop click)
  const handleOverlayClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) {
        close();
      }
    },
    [close]
  );

  if (!isOpen) return null;

  const hasResults = allItems.length > 0;

  return (
    <div className="global-search-overlay" onClick={handleOverlayClick}>
      <div className="global-search-modal" onKeyDown={handleKeyDown}>
        {/* Search Input */}
        <div className="global-search-header">
          <SearchOutlined className="global-search-icon" />
          <input
            ref={inputRef}
            type="text"
            className="global-search-input"
            placeholder="Search pages and chats..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <span className="global-search-shortcut">ESC</span>
        </div>

        {/* Search Results */}
        <div className="global-search-results">
          {hasResults ? (
            <>
              {/* Pages Section */}
              {filteredItems.pages.length > 0 && (
                <div className="global-search-section">
                  <div className="global-search-section-title">Pages</div>
                  {filteredItems.pages.map((item, idx) => {
                    const globalIndex = idx;
                    return (
                      <div
                        key={item.id}
                        className={`global-search-item ${
                          selectedIndex === globalIndex ? "selected" : ""
                        }`}
                        onClick={() => handleItemSelect(item)}
                        onMouseEnter={() => setSelectedIndex(globalIndex)}
                      >
                        <div className="global-search-item-icon">{item.icon}</div>
                        <div className="global-search-item-content">
                          <div className="global-search-item-title">{item.title}</div>
                          {item.subtitle && (
                            <div className="global-search-item-subtitle">
                              {item.subtitle}
                            </div>
                          )}
                        </div>
                        <span className="global-search-item-hint">Enter</span>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Chat History Section */}
              {filteredItems.chats.length > 0 && (
                <div className="global-search-section">
                  <div className="global-search-section-title">Recent Chats</div>
                  {filteredItems.chats.map((item, idx) => {
                    const globalIndex = filteredItems.pages.length + idx;
                    return (
                      <div
                        key={item.id}
                        className={`global-search-item ${
                          selectedIndex === globalIndex ? "selected" : ""
                        }`}
                        onClick={() => handleItemSelect(item)}
                        onMouseEnter={() => setSelectedIndex(globalIndex)}
                      >
                        <div className="global-search-item-icon">{item.icon}</div>
                        <div className="global-search-item-content">
                          <div className="global-search-item-title">{item.title}</div>
                          {item.subtitle && (
                            <div className="global-search-item-subtitle">
                              {item.subtitle}
                            </div>
                          )}
                        </div>
                        <span className="global-search-item-hint">Enter</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          ) : (
            <div className="global-search-empty">
              No results found for "{query}"
            </div>
          )}
        </div>

        {/* Footer with keyboard hints */}
        <div className="global-search-footer">
          <div className="global-search-footer-hint">
            <span className="global-search-footer-key">&uarr;</span>
            <span className="global-search-footer-key">&darr;</span>
            <span>Navigate</span>
          </div>
          <div className="global-search-footer-hint">
            <span className="global-search-footer-key">Enter</span>
            <span>Select</span>
          </div>
          <div className="global-search-footer-hint">
            <span className="global-search-footer-key">Esc</span>
            <span>Close</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper function to format relative time
function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

export { GlobalSearchProvider, useGlobalSearch } from "./global-search-context";
