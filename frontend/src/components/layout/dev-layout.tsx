import { Link, Outlet, useLocation } from "react-router-dom";
import { ArrowLeftOutlined, SearchOutlined, FileTextOutlined, ExperimentOutlined, BarChartOutlined } from "@ant-design/icons";
import { useState, useEffect } from "react";
import { LockOutlined } from "@ant-design/icons";
import { env } from "../../config/env";
import "./dev-layout.css";

const DEV_TOKEN_KEY = "dev_access_token";

const devMenuItems = [
  { path: "/dev/search", label: "Search", icon: <SearchOutlined /> },
  { path: "/dev/search-es", label: "Search ES", icon: <SearchOutlined /> },
  { path: "/dev/retrieval-test", label: "Retrieval Test", icon: <ExperimentOutlined /> },
  { path: "/dev/evaluation", label: "Evaluation", icon: <BarChartOutlined /> },
  { path: "/dev/parsing", label: "Parsing", icon: <FileTextOutlined /> },
];

export default function DevLayout() {
  const location = useLocation();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [inputToken, setInputToken] = useState("");
  const [error, setError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem(DEV_TOKEN_KEY);
    if (env.devAccessToken && stored === env.devAccessToken) {
      setIsAuthenticated(true);
    }
    setIsLoading(false);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (env.devAccessToken && inputToken === env.devAccessToken) {
      localStorage.setItem(DEV_TOKEN_KEY, inputToken);
      setIsAuthenticated(true);
      setError(false);
    } else {
      setError(true);
    }
  };

  if (isLoading) {
    return (
      <div className="dev-guard-loading">
        <div className="dev-guard-spinner" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="dev-guard-container">
        <div className="dev-guard-box">
          <div className="dev-guard-icon">
            <LockOutlined />
          </div>
          <h2 className="dev-guard-title">Developer Tools</h2>
          <p className="dev-guard-desc">
            이 페이지는 개발자 도구입니다. 접근하려면 인증이 필요합니다.
          </p>

          <form onSubmit={handleSubmit} className="dev-guard-form">
            <input
              type="password"
              value={inputToken}
              onChange={(e) => setInputToken(e.target.value)}
              placeholder="Access token"
              className={`dev-guard-input ${error ? "error" : ""}`}
            />
            {error && (
              <span className="dev-guard-error">잘못된 토큰입니다</span>
            )}
            <button type="submit" className="dev-guard-button">
              접근하기
            </button>
          </form>

          <Link to="/" className="dev-guard-back">
            메인으로 돌아가기
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="dev-layout">
      <header className="dev-header">
        <div className="dev-header-content">
          <Link to="/" className="dev-back-link">
            <ArrowLeftOutlined />
            <span>Back to Chat</span>
          </Link>
          <h1 className="dev-title">Developer Tools</h1>
          <div className="dev-spacer" />
        </div>
      </header>

      <nav className="dev-nav">
        {devMenuItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`dev-nav-item ${isActive ? "active" : ""}`}
            >
              {item.icon}
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <main className="dev-main">
        <Outlet />
      </main>
    </div>
  );
}
