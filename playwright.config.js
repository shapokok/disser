// @ts-check
const { defineConfig, devices } = require("@playwright/test");

const PORT = process.env.CROP_PORT || "5001";
const BASE_URL = process.env.BASE_URL || `http://127.0.0.1:${PORT}`;
// Locally the uv environment is used; CI installs the project into the system python.
const PYTHON = process.env.PYTHON || (require("fs").existsSync(".venv/bin/python") ? ".venv/bin/python" : "python");

module.exports = defineConfig({
  testDir: "./tests/e2e",
  timeout: 120 * 1000,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    actionTimeout: 20 * 1000,
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 5"] } },
  ],
  webServer: {
    command: `${PYTHON} backend/app.py`,
    url: `${BASE_URL}/api/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 180 * 1000,
    env: { CROP_PORT: PORT, CROP_DEVICE: process.env.CROP_DEVICE || "cpu", CROP_MODELS: process.env.CROP_MODELS === undefined ? "baseline,efficientnet,mobilenet,hybrid" : process.env.CROP_MODELS },
  },
});
