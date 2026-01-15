import { PropsWithChildren } from "react";
import "./main-content.css";

interface MainContentProps extends PropsWithChildren {
  isFullWidth?: boolean;
}

export default function MainContent({ children, isFullWidth = false }: MainContentProps) {
  return (
    <main className="main-content">
      <div className={`main-content-inner ${isFullWidth ? "full-width" : ""}`}>
        {children}
      </div>
    </main>
  );
}
