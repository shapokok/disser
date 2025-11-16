const { test, expect } = require('@playwright/test');

// Helper function to open mobile menu if needed
async function openMobileMenuIfNeeded(page) {
  const navLinks = page.locator('.nav-links');
  const isVisible = await navLinks.isVisible().catch(() => false);

  if (!isVisible) {
    // On mobile, click menu toggle to open menu
    const menuToggle = page.locator('.menu-toggle');
    if (await menuToggle.isVisible()) {
      await menuToggle.click();
      // Wait for menu to be visible and animation to complete
      await navLinks.waitFor({ state: 'visible', timeout: 5000 });
      // Extra wait for CSS transitions/animations to complete
      await page.waitForTimeout(300);
    }
  }
}

test.describe('Statistics Page', () => {
  test('should load statistics page successfully', async ({ page }) => {
    await page.goto('/frontend/stats.html');
    await expect(page).toHaveTitle(/Model Statistics - Crop Disease Detection/i);
  });

  test('should display page heading', async ({ page }) => {
    await page.goto('/frontend/stats.html');

    const heading = page.locator('h2.card-title').first();
    await expect(heading).toBeVisible();
    await expect(heading).toContainText(/Model Performance Statistics/i);
  });

  test('should display navigation menu', async ({ page }) => {
    await page.goto('/frontend/stats.html');

    // Open mobile menu if needed
    await openMobileMenuIfNeeded(page);

    // Verify nav links are present within the navigation
    const navLinks = page.locator('.nav-links');
    await expect(navLinks.getByRole('link', { name: 'Home' })).toBeVisible();
    await expect(navLinks.getByRole('link', { name: 'Analyze' })).toBeVisible();
    await expect(navLinks.getByRole('link', { name: 'Statistics' })).toBeVisible();
  });

  test('should show loading indicator or stats container', async ({ page }) => {
    await page.goto('/frontend/stats.html');

    // Either loading indicator or stats container should be present
    const loadingIndicator = page.locator('#loadingStats');
    const statsContainer = page.locator('#statsContainer');

    const loadingVisible = await loadingIndicator.isVisible().catch(() => false);
    const statsVisible = await statsContainer.isVisible().catch(() => false);

    // At least one should be present
    expect(loadingVisible || statsVisible).toBe(true);
  });

  test('should have confusion matrix model selector', async ({ page }) => {
    await page.goto('/frontend/stats.html');

    const modelSelect = page.locator('#confusionModelSelect');
    await expect(modelSelect).toBeAttached();

    // Check for model options
    const options = await modelSelect.locator('option').allTextContents();
    expect(options.some(opt => opt.includes('EfficientNet'))).toBe(true);
  });

  test('should have performance comparison section', async ({ page }) => {
    await page.goto('/frontend/stats.html');

    // Look for performance table container
    const perfTable = page.locator('#performanceTable');
    await expect(perfTable).toBeAttached();
  });

  test('should have charts container', async ({ page }) => {
    await page.goto('/frontend/stats.html');

    // Look for charts container
    const chartsContainer = page.locator('#chartsContainer');
    await expect(chartsContainer).toBeAttached();
  });

  test('should have model cards container', async ({ page }) => {
    await page.goto('/frontend/stats.html');

    // Look for model cards container
    const modelCardsContainer = page.locator('#modelCardsContainer');
    await expect(modelCardsContainer).toBeAttached();
  });

  test('should navigate back to home', async ({ page }) => {
    await page.goto('/frontend/stats.html');

    // Open mobile menu if needed
    await openMobileMenuIfNeeded(page);

    // Use href attribute for more reliable targeting in Mobile Safari
    const homeLink = page.locator('.nav-links a[href="index.html"]');
    await homeLink.click({ force: true });
    await expect(page).toHaveURL(/index\.html/);
  });
});
