// End-to-end tests for the three pages. They run against the real backend
// (started by playwright.config.js). Inference tests are skipped when no
// trained model is loaded (e.g. in CI without weights).
const { test, expect } = require("@playwright/test");
const path = require("path");

const IMAGE = path.resolve(__dirname, "../../data/test_images/test_classification.jpg");

async function health(request) {
  const res = await request.get("/api/health");
  expect(res.ok()).toBeTruthy();
  return res.json();
}

test.describe("Home page", () => {
  test("renders hero, KPIs and model cards", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/CropAI/);
    await expect(page.locator("h1")).toContainText(/диагностика|diagnosis/i);
    await expect(page.locator("#kpiClasses")).toHaveText("38");
    await expect(page.locator("#modelCards .model-card")).toHaveCount(4);
  });

  test("language switch and theme toggle work and persist", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("html")).toHaveAttribute("lang", "ru");
    await page.click('.lang-switch button[data-lang="en"]');
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await expect(page.locator("h1")).toContainText("Intelligent plant disease diagnosis");
    await page.click("#themeToggle");
    const theme = await page.locator("html").getAttribute("data-theme");
    expect(["dark", "light"]).toContain(theme);
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
  });

  test("navigation links open the other pages", async ({ page, isMobile }) => {
    await page.goto("/");
    if (isMobile) await page.click("#menuToggle");
    await page.click('.nav-links a[data-page="analyze"]');
    await expect(page).toHaveURL(/\/analyze/);
    if (isMobile) await page.click("#menuToggle");
    await page.click('.nav-links a[data-page="stats"]');
    await expect(page).toHaveURL(/\/stats/);
  });

  test("has no horizontal overflow on small screens", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 700 });
    await page.goto("/");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
    expect(overflow).toBeFalsy();
  });
});

test.describe("Analyze page", () => {
  test("settings are populated from the API", async ({ page, request }) => {
    const h = await health(request);
    await page.goto("/analyze");
    const options = page.locator("#modelSelect option");
    await expect(options).toHaveCount(h.models_loaded.length + (h.models_trained.length >= 2 ? 1 : 0));
    await expect(page.locator("#analyzeBtn")).toBeDisabled();
    await page.click('#modeSeg button[data-value="field"]');
    await expect(page.locator("#modeHint")).not.toBeEmpty();
  });

  test("accepts files, rejects wrong types", async ({ page }) => {
    await page.goto("/analyze");
    await page.setInputFiles("#fileInput", IMAGE);
    await expect(page.locator(".preview")).toHaveCount(1);
    await expect(page.locator("#analyzeBtn")).toBeEnabled();
    await page.setInputFiles("#fileInput", path.resolve(__dirname, "../../data/test_images/invalid.txt"));
    await expect(page.locator(".preview")).toHaveCount(1);
    await expect(page.locator(".toast.warning")).toBeVisible();
    await page.click(".preview .remove");
    await expect(page.locator(".preview")).toHaveCount(0);
    await expect(page.locator("#analyzeBtn")).toBeDisabled();
  });

  test("runs an analysis and shows a diagnosis with export buttons", async ({ page, request }) => {
    const h = await health(request);
    test.skip(!h.models_trained.includes("efficientnet"), "efficientnet weights are not available");
    await page.goto("/analyze");
    await page.setInputFiles("#fileInput", IMAGE);
    await page.selectOption("#modelSelect", "efficientnet");
    await page.click('#explanationSeg button[data-value="gradcam"]');
    await page.click("#analyzeBtn");
    const card = page.locator("article.result-card").first();
    await expect(card).toBeVisible({ timeout: 90 * 1000 });
    await expect(card.locator(".diagnosis .disease")).not.toBeEmpty();
    await expect(card.locator(".topk li")).toHaveCount(5);
    await expect(card.locator(".viz img")).toHaveAttribute("src", /^data:image\/jpeg/);
    await expect(page.locator("#exportBar")).toBeVisible();
    const [download] = await Promise.all([page.waitForEvent("download"), card.locator(".result-footer button", { hasText: "CSV" }).click()]);
    expect(download.suggestedFilename()).toMatch(/\.csv$/);
  });

  test("compares all loaded models", async ({ page, request }) => {
    const h = await health(request);
    test.skip(h.models_loaded.length < 2, "needs at least two models");
    await page.goto("/analyze");
    await page.setInputFiles("#fileInput", IMAGE);
    await page.click("#compareBtn");
    const card = page.locator("article.result-card").first();
    await expect(card).toBeVisible({ timeout: 90 * 1000 });
    await expect(card.locator("table.data tbody tr")).toHaveCount(h.models_loaded.length);
  });
});

test.describe("Statistics page", () => {
  test("tabs switch and tables render", async ({ page }) => {
    await page.goto("/stats");
    await expect(page.locator("#overviewTiles .tile")).toHaveCount(4);
    await expect(page.locator("#modelsTable table")).toBeVisible();
    for (const tab of ["training", "classes", "confusion", "research"]) {
      await page.click(`#tabs button[data-tab="${tab}"]`);
      await expect(page.locator(`#tab-${tab}`)).toBeVisible();
      await expect(page).toHaveURL(new RegExp(`#${tab}$`));
    }
  });

  test("per-class table and confusion matrix use the evaluation files when present", async ({ page, request }) => {
    const res = await request.get("/api/validation/all");
    const reports = (await res.json()).reports || {};
    test.skip(!Object.keys(reports).length, "no validation files (run scripts/evaluate_models.py)");
    await page.goto("/stats#classes");
    await expect(page.locator("#classTable tbody tr")).toHaveCount(38);
    await page.fill("#classSearch", "Tomato");
    await expect(page.locator("#classTable tbody tr").first()).toContainText(/Tomato|Томат/);
    await page.click('#tabs button[data-tab="confusion"]');
    await expect(page.locator(".cm-cell")).toHaveCount(38 * 38);
  });

  test("deep link opens the research tab", async ({ page }) => {
    await page.goto("/stats#research");
    await expect(page.locator("#tab-research")).toBeVisible();
    await expect(page.locator("#tab-overview")).toBeHidden();
  });
});
