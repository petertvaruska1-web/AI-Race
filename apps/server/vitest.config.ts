import { defineProject } from "vitest/config";

// Each test server opens its own PGlite database (1–3 s, more under parallel load).
export default defineProject({
  test: {
    name: "@airace/server",
    testTimeout: 30_000,
    hookTimeout: 30_000,
  },
});
