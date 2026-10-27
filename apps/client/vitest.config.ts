import react from "@vitejs/plugin-react";
import { defineProject } from "vitest/config";

export default defineProject({
  plugins: [react()],
  test: {
    name: "@airace/client",
    environment: "happy-dom",
    setupFiles: ["./test/setup.ts"],
  },
});
