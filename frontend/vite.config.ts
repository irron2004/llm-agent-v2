import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

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
    port: 4173,
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
