import React from "react";
import ReactDOM from "react-dom/client";
import AppProviders from "./app/providers";
import AppRouter from "./app/router";
import "antd/dist/reset.css";
import "./styles.css";

const container = document.getElementById("root");

if (!container) {
  throw new Error("Root container #root not found");
}

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <AppProviders>
      <AppRouter />
    </AppProviders>
  </React.StrictMode>
);
