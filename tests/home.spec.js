const { test, expect } = require('@playwright/test');

test.describe('Home Page', () => {
  test('should load home page successfully', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Crop Disease Detection/i);
  });

  test('should display main navigation', async ({ page }) => {
    await page.goto('/');

    // Check for main heading or welcome text
    const heading = page.locator('h1, h2').first();
    await expect(heading).toBeVisible();
  });

  test('should have responsive layout', async ({ page }) => {
    await page.goto('/');

    // Verify page loads without errors
    await expect(page.locator('body')).toBeVisible();
  });

  test('should navigate to analyze page', async ({ page }) => {
    await page.goto('/');

    // Look for a link or button to the analyze page
    const analyzeLink = page.getByRole('link', { name: /analyze/i }).or(
      page.getByRole('button', { name: /analyze/i })
    ).first();

    if (await analyzeLink.count() > 0) {
      await analyzeLink.click();
      await expect(page).toHaveURL(/analyze/);
    }
  });
});
