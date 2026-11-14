import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// During development the API runs separately (`uv run gaze-serve`), Vite proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": process.env.GAZE_API ?? "http://127.0.0.1:8000",
    },
  },
  build: {
    chunkSizeWarningLimit: 2000,
  },
});
