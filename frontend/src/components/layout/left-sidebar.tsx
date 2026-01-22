import { useNavigate, useLocation } from "react-router-dom";
import {
  MessageOutlined,
  SearchOutlined,
  FileTextOutlined,
  ExperimentOutlined,
  PlusOutlined,
  MenuFoldOutlined,
  DeleteOutlined,
  InboxOutlined,
} from "@ant-design/icons";
import { useState, useMemo } from "react";
import { useChatHistoryContext } from "../../features/chat/context/chat-history-context";
import { EmptyState } from "../empty-state";
import "./left-sidebar.css";

interface MenuItem {
  key: string;
  icon: React.ReactNode;
  label: string;
}

interface LeftSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export default function LeftSidebar({
  isOpen,
  onClose,
  isCollapsed = false,
  onToggleCollapse,
}: LeftSidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchQuery, setSearchQuery] = useState("");
  const { history, hideChat } = useChatHistoryContext();

  const menuItems: MenuItem[] = [
    {
      key: "/",
      icon: <MessageOutlined />,
      label: "Chat",
    },
    {
      key: "/search",
      icon: <SearchOutlined />,
      label: "Search",
    },
    {
      key: "/retrieval-test",
      icon: <ExperimentOutlined />,
      label: "Retrieval Test",
    },
    {
      key: "/parsing",
      icon: <FileTextOutlined />,
      label: "Parsing",
    },
  ];

  // Filter history by search query
  const filteredHistory = useMemo(() => {
    if (!searchQuery.trim()) return history;
    const query = searchQuery.toLowerCase();
    return history.filter(
      (item) =>
        item.title.toLowerCase().includes(query) ||
        item.preview.toLowerCase().includes(query)
    );
  }, [history, searchQuery]);

  const handleNewChat = () => {
    window.dispatchEvent(new CustomEvent("pe-agent:new-chat"));
    navigate("/");
    onClose();
  };

  const handleMenuClick = (key: string) => {
    navigate(key);
    onClose();
  };

  const handleHistoryClick = (id: string) => {
    console.log("[Sidebar] Clicking on session:", id);
    // Navigate to chat page with session parameter
    navigate(`/?session=${id}`);
    onClose();
  };

  const handleHideChat = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    hideChat(id);
  };

  const isActiveRoute = (key: string) => {
    if (key === "/") {
      return location.pathname === "/";
    }
    return location.pathname.startsWith(key);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    // Format time as HH:MM
    const timeStr = date.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", hour12: false });

    if (diffDays === 0) return `오늘 ${timeStr}`;
    if (diffDays === 1) return `어제 ${timeStr}`;
    if (diffDays < 7) return `${diffDays}일 전`;
    return date.toLocaleDateString("ko-KR");
  };

  return (
    <aside
      className={`left-sidebar ${isOpen ? "open" : ""} ${isCollapsed ? "collapsed" : ""}`}
    >
      {/* Header: Logo + New Chat */}
      <div className="sidebar-header">
        <div className="sidebar-top-row">
          <div className="sidebar-logo">
            <div className="logo-icon">RTM</div>
            <span className="logo-text">RTM Agent</span>
          </div>
          {onToggleCollapse && (
            <button
              className="sidebar-collapse-btn"
              onClick={onToggleCollapse}
              title="Collapse sidebar"
            >
              <MenuFoldOutlined />
            </button>
          )}
        </div>
        <button className="new-chat-button" onClick={handleNewChat}>
          <PlusOutlined />
          <span>New Chat</span>
        </button>
      </div>

      {/* Search */}
      <div className="sidebar-search">
        <div className="search-input-wrapper">
          <SearchOutlined className="search-icon" />
          <input
            type="text"
            placeholder="Search chats..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="sidebar-nav">
        {menuItems.map((item) => {
          const isActive = isActiveRoute(item.key);
          return (
            <button
              key={item.key}
              className={`nav-item ${isActive ? "active" : ""}`}
              onClick={() => handleMenuClick(item.key)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Chat History */}
      <div className="sidebar-history">
        <div className="history-header">Recent Chats</div>
        <div className="history-list">
          {filteredHistory.length === 0 ? (
            <EmptyState
              icon={searchQuery ? <SearchOutlined /> : <InboxOutlined />}
              title={searchQuery ? "검색 결과 없음" : "대화 내역이 없습니다"}
              description={
                searchQuery
                  ? "다른 검색어를 시도해 보세요"
                  : "새 대화를 시작해 보세요"
              }
              action={
                searchQuery
                  ? undefined
                  : { label: "새 대화 시작", onClick: handleNewChat }
              }
              size="small"
            />
          ) : (
            filteredHistory.map((item) => (
              <div
                key={item.id}
                className="history-item"
                onClick={() => handleHistoryClick(item.id)}
              >
                <MessageOutlined className="history-item-icon" />
                <div className="history-item-content">
                  <div className="history-item-title">{item.title}</div>
                  <div className="history-item-date">{formatDate(item.updatedAt)}</div>
                </div>
                <div className="history-item-actions">
                  <button
                    className="history-item-action delete"
                    onClick={(e) => handleHideChat(e, item.id)}
                    title="Hide chat"
                  >
                    <DeleteOutlined />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

    </aside>
  );
}
