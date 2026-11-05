import { defineProject } from "vitest/config";

// Each test server opens its own PGlite database (1–3 s, more under parallel load).
export default defineProject({
  test: {
    name: "@airace/server",
    testTimeout: 30_000,
    // Closing a test server (actors snapshot, the database closes) can pass 30 s while every
    // workspace's tests run at once.
    hookTimeout: 60_000,
  },
});
