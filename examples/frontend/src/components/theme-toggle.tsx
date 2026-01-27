import { MoonOutlined, SunOutlined } from "@ant-design/icons";
import { useTheme } from "./theme-provider";

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="action-button"
      title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      style={{ width: 36, height: 36 }}
    >
      {theme === "dark" ? <SunOutlined /> : <MoonOutlined />}
    </button>
  );
}
