import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      process: "process/browser",
      "@": path.resolve(__dirname, "src"),
    },
  },
  define: {
    "process.env": {},
    global: "window",
    "import.meta.env.VITE_DEBUG": JSON.stringify(process.env.VITE_DEBUG || "false"),
  },
  optimizeDeps: {
    include: [
      "@deck.gl/react",
      "@deck.gl/core",
      "@deck.gl/layers",
      "@deck.gl/geo-layers",
    ],
  },
});
