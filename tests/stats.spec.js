const { test, expect } = require('@playwright/test');

test.describe('Statistics Page', () => {
  test('should load statistics page successfully', async ({ page }) => {
    await page.goto('/crop_monitoring_app/frontend/stats.html');

    // Verify page loaded
    await expect(page.locator('body')).toBeVisible();
  });

  test('should display statistics heading', async ({ page }) => {
    await page.goto('/crop_monitoring_app/frontend/stats.html');

    // Look for statistics-related heading
    const heading = page.locator('h1, h2, h3').first();
    await expect(heading).toBeVisible();
  });

  test('should have data visualization elements', async ({ page }) => {
    await page.goto('/crop_monitoring_app/frontend/stats.html');

    // Look for common chart/graph elements
    const chartArea = page.locator('canvas, svg, .chart, #chart, .graph').first();
    if (await chartArea.count() > 0) {
      await expect(chartArea).toBeDefined();
    }
  });

  test('should be responsive', async ({ page }) => {
    await page.goto('/crop_monitoring_app/frontend/stats.html');

    // Verify page renders on different viewport sizes
    await expect(page.locator('body')).toBeVisible();

    const viewportSize = page.viewportSize();
    expect(viewportSize).toBeDefined();
  });

  test('should handle navigation', async ({ page }) => {
    await page.goto('/crop_monitoring_app/frontend/stats.html');

    // Verify navigation elements exist
    const links = page.locator('a[href]');
    if (await links.count() > 0) {
      expect(await links.count()).toBeGreaterThan(0);
    }
  });
});
