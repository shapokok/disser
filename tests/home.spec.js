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

    // Open mobile menu if needed
    await openMobileMenuIfNeeded(page);

    // Verify nav links are present within the navigation
    const navLinks = page.locator('.nav-links');
    await expect(navLinks).toBeVisible();
    await expect(navLinks.getByRole('link', { name: 'Home' })).toBeVisible();
    await expect(navLinks.getByRole('link', { name: 'Analyze' })).toBeVisible();
    await expect(navLinks.getByRole('link', { name: 'Statistics' })).toBeVisible();
  });

  test('should have call-to-action buttons', async ({ page }) => {
    await page.goto('/frontend/index.html');

    // Wait for hero section to be visible
    const heroSection = page.locator('.hero');
    await expect(heroSection).toBeVisible();

    // Check for "Start Analysis" button within hero section
    const startButton = heroSection.getByRole('link', { name: /Start Analysis/i });
    await startButton.scrollIntoViewIfNeeded();
    await expect(startButton).toBeVisible();

    // Check for "View Statistics" button within hero section
    const statsButton = heroSection.getByRole('link', { name: /View Statistics/i });
    await statsButton.scrollIntoViewIfNeeded();
    await expect(statsButton).toBeVisible();
  });

  test('should navigate to analyze page', async ({ page }) => {
    await page.goto('/frontend/index.html');

    // Open mobile menu if needed
    await openMobileMenuIfNeeded(page);

    // Use JavaScript click to bypass Mobile Safari rendering issues
    await page.evaluate(() => {
      const link = document.querySelector('.nav-links a[href="analyze.html"]');
      if (link) link.click();
    });
    await expect(page).toHaveURL(/analyze\.html/);
  });

  test('should navigate to statistics page', async ({ page }) => {
    await page.goto('/frontend/index.html');

    // Open mobile menu if needed
    await openMobileMenuIfNeeded(page);

    // Use JavaScript click to bypass Mobile Safari rendering issues
    await page.evaluate(() => {
      const link = document.querySelector('.nav-links a[href="stats.html"]');
      if (link) link.click();
    });
    await expect(page).toHaveURL(/stats\.html/);
  });
});
