import { MenuFoldOutlined } from "@ant-design/icons";
import "./right-sidebar.css";

interface RightSidebarProps {
  isOpen: boolean;
  isCollapsed?: boolean;
  onClose: () => void;
  onToggleCollapse?: () => void;
  title?: string;
  subtitle?: string;
  children?: React.ReactNode;
}

export default function RightSidebar({
  isOpen,
  isCollapsed = false,
  onClose,
  onToggleCollapse,
  title,
  subtitle,
  children,
}: RightSidebarProps) {
  if (!isOpen || isCollapsed) return null;

  return (
    <aside className="right-sidebar">
      <div className="right-sidebar-header">
        <div className="right-sidebar-title-section">
          {title && <h3 className="right-sidebar-title">{title}</h3>}
          {subtitle && <span className="right-sidebar-subtitle">{subtitle}</span>}
        </div>
        {onToggleCollapse && (
          <button
            className="right-sidebar-collapse"
            onClick={onToggleCollapse}
            title="Collapse sidebar"
          >
            <MenuFoldOutlined style={{ transform: "scaleX(-1)" }} />
          </button>
        )}
      </div>
      <div className="right-sidebar-content">{children}</div>
    </aside>
  );
}
