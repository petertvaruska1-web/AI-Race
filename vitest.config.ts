import { defineConfig } from "vitest/config";

// Every workspace folder is its own test project.
export default defineConfig({
  test: {
    projects: ["packages/*", "apps/*"],
  },
});
