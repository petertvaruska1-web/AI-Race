import { defineConfig } from "@playwright/test";

// End-to-end tests (`npm run e2e`): slow, and not part of `npm test`. They use the Microsoft
// Edge that Windows already has, so no browser download is needed.
export default defineConfig({
  testDir: "tests/e2e",
  timeout: 10 * 60_000,
  expect: { timeout: 60_000 },
  workers: 1,
  reporter: "list",
  use: {
    channel: process.env.E2E_BROWSER_CHANNEL ?? "msedge",
    trace: "retain-on-failure",
    viewport: { width: 1280, height: 900 },
  },
});
