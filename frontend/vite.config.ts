import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    // ECharts stays in an optional chunk loaded only when a real series can be rendered.
    chunkSizeWarningLimit: 550,
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
  },
});
