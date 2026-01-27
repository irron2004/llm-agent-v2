import { PropsWithChildren } from "react";
import { ThemeToggle } from "../theme-toggle";
import "./main-content.css";

interface MainContentProps extends PropsWithChildren {
  isFullWidth?: boolean;
}

export default function MainContent({ children, isFullWidth = false }: MainContentProps) {
  return (
    <main className="main-content">
      {/* Theme Toggle - Top Right of Main Content */}
      <div className="main-content-theme-toggle">
        <ThemeToggle />
      </div>
      <div className={`main-content-inner ${isFullWidth ? "full-width" : ""}`}>
        {children}
      </div>
    </main>
  );
}
