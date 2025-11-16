const { test, expect } = require('@playwright/test');

test.describe('Home Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the home page successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/Crop Disease Detection System/);
  });

  test('should display the header with navigation', async ({ page }) => {
    // Check logo
    const logo = page.locator('.logo');
    await expect(logo).toBeVisible();
    await expect(logo).toContainText('CropAI Monitor');

    // Check navigation links
    await expect(page.locator('a[href="index.html"]').first()).toBeVisible();
    await expect(page.locator('a[href="analyze.html"]').first()).toBeVisible();
    await expect(page.locator('a[href="stats.html"]').first()).toBeVisible();
  });

  test('should display hero section with main heading', async ({ page }) => {
    const heroSection = page.locator('.hero');
    await expect(heroSection).toBeVisible();

    const mainHeading = heroSection.locator('h1');
    await expect(mainHeading).toContainText('Intelligent Crop Disease Detection');
  });

  test('should have working CTA buttons in hero section', async ({ page }) => {
    const startAnalysisBtn = page.locator('.hero a[href="analyze.html"]');
    const viewStatsBtn = page.locator('.hero a[href="stats.html"]');

    await expect(startAnalysisBtn).toBeVisible();
    await expect(startAnalysisBtn).toContainText('Start Analysis');

    await expect(viewStatsBtn).toBeVisible();
    await expect(viewStatsBtn).toContainText('View Statistics');
  });

  test('should display About section', async ({ page }) => {
    const aboutSection = page.locator('.card').filter({ hasText: 'About This System' });
    await expect(aboutSection).toBeVisible();
    await expect(aboutSection).toContainText('intelligent system for monitoring');
  });

  test('should display all 6 key features', async ({ page }) => {
    const featureCards = page.locator('.feature-card');
    await expect(featureCards).toHaveCount(20); // 6 in features + 8 crops + 4 models + 2 extra

    // Check specific features exist
    await expect(page.locator('.feature-card', { hasText: 'Image Upload' })).toBeVisible();
    await expect(page.locator('.feature-card', { hasText: 'AI Detection' })).toBeVisible();
    await expect(page.locator('.feature-card', { hasText: 'Explainable AI' })).toBeVisible();
    await expect(page.locator('.feature-card', { hasText: 'Performance Metrics' })).toBeVisible();
    await expect(page.locator('.feature-card', { hasText: 'Real-time Results' })).toBeVisible();
    await expect(page.locator('.feature-card', { hasText: 'Download Results' })).toBeVisible();
  });

  test('should display supported crops section', async ({ page }) => {
    const cropsSection = page.locator('.card').filter({ hasText: 'Supported Crops & Diseases' });
    await expect(cropsSection).toBeVisible();

    // Check for specific crops
    await expect(page.locator('.feature-card', { hasText: 'Apple' })).toBeVisible();
    await expect(page.locator('.feature-card', { hasText: 'Corn' })).toBeVisible();
    await expect(page.locator('.feature-card', { hasText: 'Grape' })).toBeVisible();
    await expect(page.locator('.feature-card', { hasText: 'Tomato' })).toBeVisible();
    await expect(page.locator('.feature-card', { hasText: 'Potato' })).toBeVisible();
  });

  test('should display all 4 model architectures', async ({ page }) => {
    const modelsSection = page.locator('.card').filter({ hasText: 'Model Architectures' });
    await expect(modelsSection).toBeVisible();

    // Check all 4 models are displayed
    await expect(page.locator('.model-card', { hasText: 'Baseline CNN' })).toBeVisible();
    await expect(page.locator('.model-card', { hasText: 'EfficientNet-B0' })).toBeVisible();
    await expect(page.locator('.model-card', { hasText: 'MobileNet-V2' })).toBeVisible();
    await expect(page.locator('.model-card', { hasText: 'Hybrid CNN-Transformer' })).toBeVisible();

    // Check model stats are displayed
    const baselineModel = page.locator('.model-card', { hasText: 'Baseline CNN' });
    await expect(baselineModel.locator('.stat-label', { hasText: 'Parameters' })).toBeVisible();
    await expect(baselineModel.locator('.stat-label', { hasText: 'Size' })).toBeVisible();
    await expect(baselineModel.locator('.stat-label', { hasText: 'Speed' })).toBeVisible();
  });

  test('should display footer with correct information', async ({ page }) => {
    const footer = page.locator('.footer');
    await expect(footer).toBeVisible();
    await expect(footer).toContainText('Crop Disease Detection System');
    await expect(footer).toContainText('Master\'s Thesis Project');
    await expect(footer).toContainText('© 2025');
  });

  test('should navigate to Analyze page when clicking Start Analysis button', async ({ page }) => {
    await page.locator('.hero a[href="analyze.html"]').click();
    await expect(page).toHaveURL(/analyze\.html/);
  });

  test('should navigate to Statistics page when clicking View Statistics button', async ({ page }) => {
    await page.locator('.hero a[href="stats.html"]').click();
    await expect(page).toHaveURL(/stats\.html/);
  });

  test('should have working mobile menu toggle', async ({ page, isMobile }) => {
    if (!isMobile) {
      test.skip();
    }

    const menuToggle = page.locator('.menu-toggle');
    const navLinks = page.locator('#navLinks');

    // Menu should be hidden initially on mobile
    await expect(menuToggle).toBeVisible();

    // Click to open menu
    await menuToggle.click();
    await expect(navLinks).toHaveClass(/active/);

    // Click again to close
    await menuToggle.click();
    await expect(navLinks).not.toHaveClass(/active/);
  });

  test('should be responsive on mobile devices', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });

    // Check that main elements are still visible
    await expect(page.locator('.logo')).toBeVisible();
    await expect(page.locator('.hero h1')).toBeVisible();
    await expect(page.locator('.footer')).toBeVisible();
  });

  test('should have proper meta tags', async ({ page }) => {
    const viewport = await page.locator('meta[name="viewport"]');
    await expect(viewport).toHaveAttribute('content', 'width=device-width, initial-scale=1.0');

    const charset = await page.locator('meta[charset]');
    await expect(charset).toHaveAttribute('charset', 'UTF-8');
  });

  test('should load CSS stylesheet', async ({ page }) => {
    const stylesheetLink = page.locator('link[href="style.css"]');
    await expect(stylesheetLink).toHaveAttribute('rel', 'stylesheet');
  });

  test('should have accessible navigation', async ({ page }) => {
    // Check that navigation is keyboard accessible
    await page.keyboard.press('Tab');
    const focusedElement = await page.evaluate(() => document.activeElement.textContent);
    expect(focusedElement).toBeTruthy();
  });
});
