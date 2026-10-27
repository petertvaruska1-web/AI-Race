import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The game server to proxy to; the end-to-end tests run their own on another port.
const SERVER = process.env.AIRACE_SERVER ?? "http://127.0.0.1:8787";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": SERVER,
      "/blobs": SERVER,
      "/ws": { target: SERVER, ws: true },
    },
  },
});
