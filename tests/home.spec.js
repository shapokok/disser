const { test, expect } = require('@playwright/test');

test.describe('Home Page', () => {
  test('should load home page successfully', async ({ page }) => {
    await page.goto('/frontend/index.html');
    await expect(page).toHaveTitle(/Crop Disease Detection/i);
  });

  test('should display main heading', async ({ page }) => {
    await page.goto('/frontend/index.html');

    // Check for the main hero heading
    const heading = page.locator('h1');
    await expect(heading).toBeVisible();
    await expect(heading).toContainText(/Intelligent Crop Disease Detection/i);
  });

  test('should display navigation menu', async ({ page }) => {
    await page.goto('/frontend/index.html');

    // Verify nav links are present
    await expect(page.locator('.nav-links')).toBeVisible();
    await expect(page.getByRole('link', { name: 'Home' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Analyze' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Statistics' })).toBeVisible();
  });

  test('should have call-to-action buttons', async ({ page }) => {
    await page.goto('/frontend/index.html');

    // Check for "Start Analysis" button
    const startButton = page.getByRole('link', { name: /Start Analysis/i });
    await expect(startButton).toBeVisible();

    // Check for "View Statistics" button
    const statsButton = page.getByRole('link', { name: /View Statistics/i });
    await expect(statsButton).toBeVisible();
  });

  test('should navigate to analyze page', async ({ page }) => {
    await page.goto('/frontend/index.html');

    // Click the "Analyze" nav link
    await page.getByRole('link', { name: 'Analyze' }).click();
    await expect(page).toHaveURL(/analyze\.html/);
  });

  test('should navigate to statistics page', async ({ page }) => {
    await page.goto('/frontend/index.html');

    // Click the "Statistics" nav link
    await page.getByRole('link', { name: 'Statistics' }).click();
    await expect(page).toHaveURL(/stats\.html/);
  });
});
