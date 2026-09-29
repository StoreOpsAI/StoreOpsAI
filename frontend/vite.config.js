import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
      "/detector": {
        target:
          process.env.VITE_DETECTOR_PROXY_TARGET || "http://localhost:8100",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/detector/, ""),
      },
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
  },
});
