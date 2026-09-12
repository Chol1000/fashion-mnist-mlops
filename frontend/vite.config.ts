import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard talks to FastAPI over the same root-level paths the API has
// always exposed (/health, /predict, /metrics …) rather than an /api prefix,
// so the load tests and the documented endpoint list stay valid. In dev those
// paths have to be proxied one by one — a blanket "/" proxy would swallow
// Vite's own module and HMR requests.
const API_PATHS = [
  "/health",
  "/predict",
  "/upload-data",
  "/retrain",
  "/uploaded-data",
  "/metrics",
  "/insights",
  "/sample",
  "/figures",
  "/docs",
  "/openapi.json",
];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    allowedHosts: true,
    proxy: Object.fromEntries(
      API_PATHS.map((p) => [
        p,
        {
          // Overridable because 8000 is a common default another local project
          // may have taken — point VITE_API_TARGET at wherever uvicorn is bound
          // rather than editing this file.
          target: process.env.VITE_API_TARGET || "http://127.0.0.1:8000",
          changeOrigin: true,
        },
      ])
    ),
  },
  build: {
    // The Space runs on a shared free-tier CPU; splitting the heavy libraries
    // out keeps the first paint (shell + Overview) from waiting on recharts,
    // which only three of the seven pages need.
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("recharts") || id.includes("d3-")) return "charts";
          if (id.includes("antd") || id.includes("@ant-design") || id.includes("rc-")) return "antd";
          return undefined;
        },
      },
    },
  },
});
