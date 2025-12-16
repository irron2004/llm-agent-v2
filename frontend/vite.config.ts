import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const API_PROXY_TARGET = process.env.API_PROXY_TARGET || "http://localhost:8000";
const PORT = Number(process.env.PORT || process.env.FRONTEND_PORT || 9097);

// 프로젝트 루트의 data 폴더를 /data 경로로 서빙하는 플러그인
function serveDataFolder(): Plugin {
  const dataDir = resolve(__dirname, "../data");
  return {
    name: "serve-data-folder",
    configureServer(server) {
      server.middlewares.use("/data", (req, res, next) => {
        const filePath = resolve(dataDir, decodeURIComponent(req.url || "").slice(1));
        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
          res.setHeader("Access-Control-Allow-Origin", "*");
          const stream = fs.createReadStream(filePath);
          stream.pipe(res);
        } else {
          next();
        }
      });
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), serveDataFolder()],
  server: {
    port: PORT,
    proxy: {
      // Frontend dev 서버에서 /api/* 요청을 백엔드로 프록시
      "/api": {
        target: API_PROXY_TARGET,
        changeOrigin: true,
      },
    },
    fs: {
      allow: [".", "../data"],
    },
  },
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
    },
  },
});
