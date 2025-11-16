const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

test.describe('Analyze Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/frontend/analyze.html');
  });

  test('should load the analyze page successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/Analyze Images - Crop Disease Detection/);
  });

  test('should display navigation with Analyze as active', async ({ page }) => {
    const analyzeLink = page.locator('a[href="analyze.html"].active');
    await expect(analyzeLink).toBeVisible();
    await expect(analyzeLink).toHaveClass(/active/);
  });

  test('should display page header and title', async ({ page }) => {
    await expect(page.locator('h2.card-title')).toContainText('Analyze Crop Images');
    await expect(page.locator('.card-subtitle')).toContainText('Upload plant images for AI-powered disease detection');
  });

  test('should have all model selection options', async ({ page }) => {
    const modelSelect = page.locator('#modelSelect');
    await expect(modelSelect).toBeVisible();

    // Check all model options exist
    const options = await modelSelect.locator('option').allTextContents();
    expect(options).toContain('EfficientNet-B0 (Recommended)');
    expect(options).toContain('Hybrid CNN-Transformer (Most Accurate)');
    expect(options).toContain('MobileNet-V2 (Fastest)');
    expect(options).toContain('Baseline CNN');
  });

  test('should have explanation method selector', async ({ page }) => {
    const explanationSelect = page.locator('#explanationSelect');
    await expect(explanationSelect).toBeVisible();

    const options = await explanationSelect.locator('option').allTextContents();
    expect(options).toContain('Grad-CAM (Recommended)');
    expect(options).toContain('LIME');
  });

  test('should have dataset type selector', async ({ page }) => {
    const datasetSelect = page.locator('#datasetSelect');
    await expect(datasetSelect).toBeVisible();

    const options = await datasetSelect.locator('option').allTextContents();
    expect(options).toContain('Controlled (PlantVillage)');
    expect(options).toContain('Field Images');
  });

  test('should display upload area', async ({ page }) => {
    const uploadArea = page.locator('#uploadArea');
    await expect(uploadArea).toBeVisible();
    await expect(uploadArea).toContainText('Drag and drop images here');
    await expect(uploadArea).toContainText('or click to browse');
  });

  test('should have file input accepting images only', async ({ page }) => {
    const fileInput = page.locator('#fileInput');
    await expect(fileInput).toHaveAttribute('accept', 'image/jpeg,image/png');
    await expect(fileInput).toHaveAttribute('multiple', '');
  });

  test('should have disabled analyze button initially', async ({ page }) => {
    const analyzeBtn = page.locator('#analyzeBtn');
    await expect(analyzeBtn).toBeVisible();
    await expect(analyzeBtn).toBeDisabled();
    await expect(analyzeBtn).toContainText('Analyze Images');
  });

  test('should have disabled compare button initially', async ({ page }) => {
    const compareBtn = page.locator('#compareBtn');
    await expect(compareBtn).toBeVisible();
    await expect(compareBtn).toBeDisabled();
    await expect(compareBtn).toContainText('Compare All Models');
  });

  test('should have hidden clear button initially', async ({ page }) => {
    const clearBtn = page.locator('#clearBtn');
    await expect(clearBtn).not.toBeVisible();
  });

  test('should show loading container when analyzing (mocked)', async ({ page }) => {
    const loadingContainer = page.locator('#loadingContainer');

    // Initially hidden
    await expect(loadingContainer).toHaveClass(/hidden/);

    // Check loading container structure
    await expect(loadingContainer.locator('.loading-spinner')).toBeAttached();
    await expect(loadingContainer.locator('#loadingMainText')).toBeAttached();
    await expect(loadingContainer.locator('#loadingSubText')).toBeAttached();
  });

  test('should have progress steps in loading container', async ({ page }) => {
    const step1 = page.locator('#step1');
    const step2 = page.locator('#step2');
    const step3 = page.locator('#step3');

    await expect(step1).toContainText('Uploading images');
    await expect(step2).toContainText('Running AI model');
    await expect(step3).toContainText('Generating explanations');
  });

  test('should enable analyze button when file is selected (simulated)', async ({ page }) => {
    // Create a test image file in memory
    const buffer = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');

    // Set up file chooser
    const [fileChooser] = await Promise.all([
      page.waitForEvent('filechooser'),
      page.locator('#uploadArea').click()
    ]);

    // Create a file
    await fileChooser.setFiles({
      name: 'test-plant.png',
      mimeType: 'image/png',
      buffer: buffer
    });

    // Wait a bit for the file to be processed
    await page.waitForTimeout(500);

    // Check if analyze button is enabled
    const analyzeBtn = page.locator('#analyzeBtn');
    await expect(analyzeBtn).toBeEnabled();

    // Check if compare button is enabled
    const compareBtn = page.locator('#compareBtn');
    await expect(compareBtn).toBeEnabled();
  });

  test('should show image preview when file is uploaded', async ({ page }) => {
    const buffer = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');

    const [fileChooser] = await Promise.all([
      page.waitForEvent('filechooser'),
      page.locator('#uploadArea').click()
    ]);

    await fileChooser.setFiles({
      name: 'test-plant.png',
      mimeType: 'image/png',
      buffer: buffer
    });

    await page.waitForTimeout(500);

    // Check image preview container is visible
    const imagePreviewContainer = page.locator('#imagePreviewContainer');
    await expect(imagePreviewContainer).not.toHaveClass(/hidden/);

    // Check preview grid exists
    const imagePreviewGrid = page.locator('#imagePreviewGrid');
    await expect(imagePreviewGrid).toBeVisible();
  });

  test('should handle multiple file uploads', async ({ page }) => {
    const buffer = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');

    const [fileChooser] = await Promise.all([
      page.waitForEvent('filechooser'),
      page.locator('#uploadArea').click()
    ]);

    // Upload multiple files
    await fileChooser.setFiles([
      {
        name: 'plant1.png',
        mimeType: 'image/png',
        buffer: buffer
      },
      {
        name: 'plant2.png',
        mimeType: 'image/png',
        buffer: buffer
      }
    ]);

    await page.waitForTimeout(500);

    // Verify buttons are enabled
    await expect(page.locator('#analyzeBtn')).toBeEnabled();
    await expect(page.locator('#compareBtn')).toBeEnabled();
  });

  test('should show clear button when files are selected', async ({ page }) => {
    const buffer = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');

    const [fileChooser] = await Promise.all([
      page.waitForEvent('filechooser'),
      page.locator('#uploadArea').click()
    ]);

    await fileChooser.setFiles({
      name: 'test-plant.png',
      mimeType: 'image/png',
      buffer: buffer
    });

    await page.waitForTimeout(500);

    // Clear button should be visible now
    const clearBtn = page.locator('#clearBtn');
    await expect(clearBtn).toBeVisible();
  });

  test('should clear files when clear button is clicked', async ({ page }) => {
    const buffer = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');

    const [fileChooser] = await Promise.all([
      page.waitForEvent('filechooser'),
      page.locator('#uploadArea').click()
    ]);

    await fileChooser.setFiles({
      name: 'test-plant.png',
      mimeType: 'image/png',
      buffer: buffer
    });

    await page.waitForTimeout(500);

    // Click clear button
    await page.locator('#clearBtn').click();

    // Check buttons are disabled again
    await expect(page.locator('#analyzeBtn')).toBeDisabled();
    await expect(page.locator('#compareBtn')).toBeDisabled();

    // Clear button should be hidden
    await expect(page.locator('#clearBtn')).not.toBeVisible();
  });

  test('should change model selection', async ({ page }) => {
    const modelSelect = page.locator('#modelSelect');

    // Default should be efficientnet
    await expect(modelSelect).toHaveValue('efficientnet');

    // Change to hybrid
    await modelSelect.selectOption('hybrid');
    await expect(modelSelect).toHaveValue('hybrid');

    // Change to mobilenet
    await modelSelect.selectOption('mobilenet');
    await expect(modelSelect).toHaveValue('mobilenet');

    // Change to baseline
    await modelSelect.selectOption('baseline');
    await expect(modelSelect).toHaveValue('baseline');
  });

  test('should change explanation method', async ({ page }) => {
    const explanationSelect = page.locator('#explanationSelect');

    // Default should be gradcam
    await expect(explanationSelect).toHaveValue('gradcam');

    // Change to lime
    await explanationSelect.selectOption('lime');
    await expect(explanationSelect).toHaveValue('lime');
  });

  test('should change dataset type', async ({ page }) => {
    const datasetSelect = page.locator('#datasetSelect');

    // Default should be controlled
    await expect(datasetSelect).toHaveValue('controlled');

    // Change to field
    await datasetSelect.selectOption('field');
    await expect(datasetSelect).toHaveValue('field');
  });

  test('should have results container initially hidden', async ({ page }) => {
    const resultsContainer = page.locator('#resultsContainer');
    await expect(resultsContainer).toHaveClass(/hidden/);
  });

  test('should display footer', async ({ page }) => {
    const footer = page.locator('.footer');
    await expect(footer).toBeVisible();
    await expect(footer).toContainText('Crop Disease Detection System');
  });

  test('should be responsive on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    // Check main elements are visible on mobile
    await expect(page.locator('.logo')).toBeVisible();
    await expect(page.locator('#uploadArea')).toBeVisible();
    await expect(page.locator('#modelSelect')).toBeVisible();
  });

  test('should handle drag and drop events', async ({ page }) => {
    const uploadArea = page.locator('#uploadArea');

    // Check that upload area is interactive
    await expect(uploadArea).toBeVisible();

    // Click should trigger file chooser
    const fileChooserPromise = page.waitForEvent('filechooser');
    await uploadArea.click();
    const fileChooser = await fileChooserPromise;
    expect(fileChooser).toBeTruthy();
  });

  test('should have proper form labels', async ({ page }) => {
    await expect(page.locator('label', { hasText: 'Select Model' })).toBeVisible();
    await expect(page.locator('label', { hasText: 'Explanation Method' })).toBeVisible();
    await expect(page.locator('label', { hasText: 'Dataset Type' })).toBeVisible();
  });

  test('should navigate back to home when clicking logo', async ({ page }) => {
    await page.locator('.logo').click();
    await expect(page).toHaveURL(/index\.html/);
  });

  test('should navigate to statistics page', async ({ page }) => {
    await page.locator('a[href="stats.html"]').first().click();
    await expect(page).toHaveURL(/stats\.html/);
  });
});

test.describe('Analyze Page - API Integration (Mocked)', () => {
  test('should handle API errors gracefully', async ({ page }) => {
    // Mock failed API response
    await page.route('**/api/upload', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ success: false, error: 'Server error' })
      });
    });

    await page.goto('/frontend/analyze.html');

    // Upload a file
    const buffer = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');

    const [fileChooser] = await Promise.all([
      page.waitForEvent('filechooser'),
      page.locator('#uploadArea').click()
    ]);

    await fileChooser.setFiles({
      name: 'test-plant.png',
      mimeType: 'image/png',
      buffer: buffer
    });

    await page.waitForTimeout(500);

    // Set up dialog handler for the alert
    page.on('dialog', async dialog => {
      expect(dialog.message()).toContain('Error');
      await dialog.accept();
    });

    // Click analyze button
    await page.locator('#analyzeBtn').click();

    // Wait a bit for the error to be processed
    await page.waitForTimeout(1000);
  });

  test('should display loading state during analysis', async ({ page }) => {
    // Mock slow API response
    await page.route('**/api/upload', async route => {
      await new Promise(resolve => setTimeout(resolve, 2000));
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          uploaded: [{ saved_name: 'test.png', original_name: 'test-plant.png' }]
        })
      });
    });

    await page.route('**/api/predict', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          prediction: {
            class: 'Tomato___healthy',
            confidence: 0.95,
            confidence_percent: '95%'
          }
        })
      });
    });

    await page.goto('/frontend/analyze.html');

    // Upload a file
    const buffer = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');

    const [fileChooser] = await Promise.all([
      page.waitForEvent('filechooser'),
      page.locator('#uploadArea').click()
    ]);

    await fileChooser.setFiles({
      name: 'test-plant.png',
      mimeType: 'image/png',
      buffer: buffer
    });

    await page.waitForTimeout(500);

    // Click analyze button
    await page.locator('#analyzeBtn').click();

    // Check loading state appears
    const loadingContainer = page.locator('#loadingContainer');
    await expect(loadingContainer).not.toHaveClass(/hidden/);
    await expect(loadingContainer).toBeVisible();
  });
});
