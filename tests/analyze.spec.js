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
      // Wait for menu to be visible
      await navLinks.waitFor({ state: 'visible', timeout: 5000 });
    }
  }
}

test.describe('Analyze Page', () => {
  test('should load analyze page successfully', async ({ page }) => {
    await page.goto('/frontend/analyze.html');
    await expect(page).toHaveTitle(/Analyze Images - Crop Disease Detection/i);
  });

  test('should display page heading', async ({ page }) => {
    await page.goto('/frontend/analyze.html');

    const heading = page.locator('h2.card-title');
    await expect(heading).toBeVisible();
    await expect(heading).toContainText(/Analyze Crop Images/i);
  });

  test('should display model selection dropdown', async ({ page }) => {
    await page.goto('/frontend/analyze.html');

    const modelSelect = page.locator('#modelSelect');
    await expect(modelSelect).toBeVisible();

    // Check for model options
    await expect(modelSelect).toContainText(/EfficientNet/i);
    await expect(modelSelect).toContainText(/MobileNet/i);
  });

  test('should display explanation method selector', async ({ page }) => {
    await page.goto('/frontend/analyze.html');

    const explanationSelect = page.locator('#explanationSelect');
    await expect(explanationSelect).toBeVisible();

    // Check for explanation options
    await expect(explanationSelect).toContainText(/Grad-CAM/i);
    await expect(explanationSelect).toContainText(/LIME/i);
  });

  test('should display dataset type selector', async ({ page }) => {
    await page.goto('/frontend/analyze.html');

    const datasetSelect = page.locator('#datasetSelect');
    await expect(datasetSelect).toBeVisible();

    // Check for dataset options
    await expect(datasetSelect).toContainText(/Controlled/i);
    await expect(datasetSelect).toContainText(/Field/i);
  });

  test('should display upload area', async ({ page }) => {
    await page.goto('/frontend/analyze.html');

    const uploadArea = page.locator('#uploadArea');
    await expect(uploadArea).toBeVisible();

    // Check upload text
    await expect(uploadArea).toContainText(/Drag and drop/i);
  });

  test('should have file input element', async ({ page }) => {
    await page.goto('/frontend/analyze.html');

    const fileInput = page.locator('#fileInput');
    await expect(fileInput).toBeAttached();

    // Verify it accepts the right file types
    await expect(fileInput).toHaveAttribute('accept', /image/i);
  });

  test('should display navigation menu', async ({ page }) => {
    await page.goto('/frontend/analyze.html');

    // Open mobile menu if needed
    await openMobileMenuIfNeeded(page);

    // Verify nav links are present within the navigation
    const navLinks = page.locator('.nav-links');
    await expect(navLinks.getByRole('link', { name: 'Home' })).toBeVisible();
    await expect(navLinks.getByRole('link', { name: 'Analyze' })).toBeVisible();
    await expect(navLinks.getByRole('link', { name: 'Statistics' })).toBeVisible();
  });
});
