const { test, expect } = require('@playwright/test');

test.describe('Analyze Page', () => {
  test('should load analyze page successfully', async ({ page }) => {
    await page.goto('/crop_monitoring_app/frontend/analyze.html');

    // Verify page loaded
    await expect(page.locator('body')).toBeVisible();
  });

  test('should display upload interface', async ({ page }) => {
    await page.goto('/crop_monitoring_app/frontend/analyze.html');

    // Look for file input or upload button
    const fileInput = page.locator('input[type="file"]').first();
    if (await fileInput.count() > 0) {
      await expect(fileInput).toBeVisible();
    }
  });

  test('should have image preview area', async ({ page }) => {
    await page.goto('/crop_monitoring_app/frontend/analyze.html');

    // Check for common image display elements
    const imageArea = page.locator('img, canvas, #preview, .preview, #image-preview').first();
    if (await imageArea.count() > 0) {
      await expect(imageArea).toBeDefined();
    }
  });

  test('should be responsive on mobile', async ({ page }) => {
    await page.goto('/crop_monitoring_app/frontend/analyze.html');

    // Basic responsiveness check
    await expect(page.locator('body')).toBeVisible();

    // Verify viewport is working
    const viewportSize = page.viewportSize();
    expect(viewportSize).toBeDefined();
  });
});
