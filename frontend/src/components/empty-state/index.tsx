import { ReactNode } from "react";
import "./empty-state.css";

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  size?: "small" | "medium" | "large";
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  size = "medium",
}: EmptyStateProps) {
  return (
    <div className={`empty-state empty-state--${size}`}>
      {icon && <div className="empty-state-icon">{icon}</div>}
      <h3 className="empty-state-title">{title}</h3>
      {description && <p className="empty-state-description">{description}</p>}
      {action && (
        <button className="empty-state-action" onClick={action.onClick}>
          {action.label}
        </button>
      )}
    </div>
  );
}
