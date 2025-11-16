const { test, expect } = require('@playwright/test');

test.describe('Statistics Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/frontend/stats.html');
  });

  test('should load the statistics page successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/Model Statistics - Crop Disease Detection/);
  });

  test('should display navigation with Statistics as active', async ({ page }) => {
    const statsLink = page.locator('a[href="stats.html"].active');
    await expect(statsLink).toBeVisible();
    await expect(statsLink).toHaveClass(/active/);
  });

  test('should display page header', async ({ page }) => {
    await expect(page.locator('h2.card-title').first()).toContainText('Model Performance Statistics');
    await expect(page.locator('.card-subtitle').first()).toContainText('Comparative analysis');
  });

  test('should show loading indicator initially', async ({ page }) => {
    const loadingStats = page.locator('#loadingStats');
    await expect(loadingStats).toBeVisible();
    await expect(loadingStats).toContainText('Loading model statistics');
  });

  test('should have stats container hidden initially', async ({ page }) => {
    const statsContainer = page.locator('#statsContainer');
    await expect(statsContainer).toHaveClass(/hidden/);
  });

  test('should display dataset information section', async ({ page }) => {
    await expect(page.locator('text=Dataset Information')).toBeAttached();
    await expect(page.locator('text=38').first()).toBeAttached(); // Disease Classes
    await expect(page.locator('text=14')).toBeAttached(); // Plant Types
    await expect(page.locator('text=54,306')).toBeAttached(); // Training Images
    await expect(page.locator('text=4').first()).toBeAttached(); // Model Architectures
  });

  test('should display model recommendations section', async ({ page }) => {
    await expect(page.locator('text=Model Recommendations')).toBeAttached();
    await expect(page.locator('text=Best Accuracy')).toBeAttached();
    await expect(page.locator('text=Best Speed')).toBeAttached();
    await expect(page.locator('text=Best Balance')).toBeAttached();
  });

  test('should show Hybrid CNN-Transformer as best accuracy', async ({ page }) => {
    const bestAccuracyCard = page.locator('.feature-card', { has: page.locator('text=Best Accuracy') });
    await expect(bestAccuracyCard).toContainText('Hybrid CNN-Transformer');
    await expect(bestAccuracyCard).toContainText('96.7% accuracy');
  });

  test('should show MobileNet-V2 as fastest', async ({ page }) => {
    const bestSpeedCard = page.locator('.feature-card', { has: page.locator('text=Best Speed') });
    await expect(bestSpeedCard).toContainText('MobileNet-V2');
    await expect(bestSpeedCard).toContainText('32ms inference');
  });

  test('should show EfficientNet-B0 as best balance', async ({ page }) => {
    const bestBalanceCard = page.locator('.feature-card', { has: page.locator('text=Best Balance') });
    await expect(bestBalanceCard).toContainText('EfficientNet-B0');
    await expect(bestBalanceCard).toContainText('95.4% accuracy');
  });

  test('should display training configuration section', async ({ page }) => {
    await expect(page.locator('text=Training Configuration')).toBeAttached();
    await expect(page.locator('text=Optimizer:')).toBeAttached();
    await expect(page.locator('text=Adam')).toBeAttached();
    await expect(page.locator('text=Learning Rate:')).toBeAttached();
    await expect(page.locator('text=0.001')).toBeAttached();
    await expect(page.locator('text=Batch Size:')).toBeAttached();
    await expect(page.locator('text=32').first()).toBeAttached();
    await expect(page.locator('text=Epochs:')).toBeAttached();
    await expect(page.locator('text=50')).toBeAttached();
    await expect(page.locator('text=Input Size:')).toBeAttached();
    await expect(page.locator('text=224x224')).toBeAttached();
  });

  test('should have confusion matrix selector', async ({ page }) => {
    const confusionModelSelect = page.locator('#confusionModelSelect');
    await expect(confusionModelSelect).toBeAttached();

    const options = await confusionModelSelect.locator('option').allTextContents();
    expect(options).toContain('EfficientNet-B0');
    expect(options).toContain('Hybrid CNN-Transformer');
    expect(options).toContain('MobileNet-V2');
    expect(options).toContain('Baseline CNN');
  });

  test('should have confusion matrix container', async ({ page }) => {
    const container = page.locator('#confusionMatrixContainer');
    await expect(container).toBeAttached();
  });

  test('should have confused pairs container', async ({ page }) => {
    const container = page.locator('#confusedPairsContainer');
    await expect(container).toBeAttached();
  });

  test('should have per-class performance section', async ({ page }) => {
    const perClassSection = page.locator('#perClassPerformance');
    await expect(perClassSection).toBeAttached();
  });

  test('should display footer', async ({ page }) => {
    const footer = page.locator('.footer');
    await expect(footer).toBeVisible();
    await expect(footer).toContainText('Crop Disease Detection System');
  });

  test('should navigate to home page', async ({ page }) => {
    await page.locator('a[href="index.html"]').first().click();
    await expect(page).toHaveURL(/index\.html/);
  });

  test('should navigate to analyze page', async ({ page }) => {
    await page.locator('a[href="analyze.html"]').first().click();
    await expect(page).toHaveURL(/analyze\.html/);
  });

  test('should be responsive on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    await expect(page.locator('.logo')).toBeVisible();
    await expect(page.locator('h2.card-title').first()).toBeVisible();
  });
});

test.describe('Statistics Page - API Integration (Mocked)', () => {
  test('should load and display statistics when API returns data', async ({ page }) => {
    // Mock successful stats API response
    await page.route('**/api/stats', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          total_classes: 38,
          statistics: {
            'baseline': {
              accuracy: 0.923,
              precision: 0.925,
              recall: 0.921,
              f1_score: 0.923,
              inference_time_ms: 45,
              parameters: '1.2M',
              size_mb: '4.8'
            },
            'efficientnet': {
              accuracy: 0.954,
              precision: 0.956,
              recall: 0.953,
              f1_score: 0.954,
              inference_time_ms: 58,
              parameters: '4.0M',
              size_mb: '16.2'
            },
            'mobilenet': {
              accuracy: 0.941,
              precision: 0.943,
              recall: 0.939,
              f1_score: 0.941,
              inference_time_ms: 32,
              parameters: '2.3M',
              size_mb: '9.1'
            },
            'hybrid': {
              accuracy: 0.967,
              precision: 0.968,
              recall: 0.966,
              f1_score: 0.967,
              inference_time_ms: 125,
              parameters: '25.6M',
              size_mb: '102.4'
            }
          }
        })
      });
    });

    await page.goto('/frontend/stats.html');

    // Wait for stats to load
    await page.waitForTimeout(1000);

    // Check that loading is hidden
    const loadingStats = page.locator('#loadingStats');
    await expect(loadingStats).toHaveClass(/hidden/);

    // Check that stats container is visible
    const statsContainer = page.locator('#statsContainer');
    await expect(statsContainer).not.toHaveClass(/hidden/);

    // Check that performance table is populated
    const performanceTable = page.locator('#performanceTable');
    await expect(performanceTable).toBeVisible();
    await expect(performanceTable).toContainText('Baseline CNN');
    await expect(performanceTable).toContainText('EfficientNet-B0');
    await expect(performanceTable).toContainText('MobileNet-V2');
    await expect(performanceTable).toContainText('Hybrid CNN-Transformer');

    // Check accuracy values
    await expect(performanceTable).toContainText('92.3%'); // Baseline
    await expect(performanceTable).toContainText('95.4%'); // EfficientNet
    await expect(performanceTable).toContainText('94.1%'); // MobileNet
    await expect(performanceTable).toContainText('96.7%'); // Hybrid
  });

  test('should display model cards when statistics load', async ({ page }) => {
    await page.route('**/api/stats', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          total_classes: 38,
          statistics: {
            'efficientnet': {
              accuracy: 0.954,
              precision: 0.956,
              recall: 0.953,
              f1_score: 0.954,
              inference_time_ms: 58,
              parameters: '4.0M',
              size_mb: '16.2'
            }
          }
        })
      });
    });

    await page.goto('/frontend/stats.html');
    await page.waitForTimeout(1000);

    // Check model cards are displayed
    const modelCardsContainer = page.locator('#modelCardsContainer');
    await expect(modelCardsContainer).toBeVisible();

    const modelCard = modelCardsContainer.locator('.model-card').first();
    await expect(modelCard).toBeVisible();
    await expect(modelCard).toContainText('EfficientNet-B0');
    await expect(modelCard).toContainText('95.40%');
  });

  test('should display charts when statistics load', async ({ page }) => {
    await page.route('**/api/stats', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          total_classes: 38,
          statistics: {
            'baseline': {
              accuracy: 0.923,
              precision: 0.925,
              recall: 0.921,
              f1_score: 0.923,
              inference_time_ms: 45,
              parameters: '1.2M',
              size_mb: '4.8'
            }
          }
        })
      });
    });

    await page.goto('/frontend/stats.html');
    await page.waitForTimeout(1000);

    // Check charts are displayed
    const chartsContainer = page.locator('#chartsContainer');
    await expect(chartsContainer).toBeVisible();
    await expect(chartsContainer).toContainText('Accuracy Comparison');
    await expect(chartsContainer).toContainText('Inference Speed Comparison');
    await expect(chartsContainer).toContainText('F1-Score Comparison');

    // Check progress bars are present
    const progressBars = page.locator('.progress-bar');
    expect(await progressBars.count()).toBeGreaterThan(0);
  });

  test('should handle API error gracefully', async ({ page }) => {
    // Mock failed API response
    await page.route('**/api/stats', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ success: false, error: 'Server error' })
      });
    });

    await page.goto('/frontend/stats.html');
    await page.waitForTimeout(1000);

    // Check error message is displayed
    const loadingStats = page.locator('#loadingStats');
    await expect(loadingStats).toContainText('Error');
    await expect(loadingStats).toContainText('Could not load statistics');
  });

  test('should load confusion matrix when model is selected', async ({ page }) => {
    // Mock stats API
    await page.route('**/api/stats', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          total_classes: 38,
          statistics: {
            'efficientnet': {
              accuracy: 0.954,
              precision: 0.956,
              recall: 0.953,
              f1_score: 0.954,
              inference_time_ms: 58,
              parameters: '4.0M',
              size_mb: '16.2'
            }
          }
        })
      });
    });

    // Mock confusion matrix API
    await page.route('**/api/confusion_matrix/efficientnet', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          confusion_matrix_image: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          confusion_matrix_normalized_image: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          top_confused_pairs: [
            {
              true_class: 'Tomato___Early_Blight',
              predicted_class: 'Tomato___Late_Blight',
              count: 15,
              percentage: 3.2
            }
          ]
        })
      });
    });

    // Mock validation API
    await page.route('**/api/validation/efficientnet', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          report: {
            per_class_metrics: [
              {
                class_name: 'Tomato___healthy',
                precision: 0.99,
                recall: 0.98,
                f1_score: 0.985,
                support: 100
              }
            ],
            best_classes: [
              {
                class_name: 'Tomato___healthy',
                precision: 0.99,
                recall: 0.98,
                f1_score: 0.985
              }
            ],
            worst_classes: [
              {
                class_name: 'Tomato___Early_Blight',
                precision: 0.85,
                recall: 0.82,
                f1_score: 0.835
              }
            ]
          }
        })
      });
    });

    await page.goto('/frontend/stats.html');
    await page.waitForTimeout(2000);

    // Check that confusion matrix images are displayed
    const confusionMatrixContainer = page.locator('#confusionMatrixContainer');
    await expect(confusionMatrixContainer.locator('img').first()).toBeVisible();
  });

  test('should change confusion matrix when model selection changes', async ({ page }) => {
    // Mock stats API
    await page.route('**/api/stats', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          total_classes: 38,
          statistics: {
            'efficientnet': {
              accuracy: 0.954,
              precision: 0.956,
              recall: 0.953,
              f1_score: 0.954,
              inference_time_ms: 58,
              parameters: '4.0M',
              size_mb: '16.2'
            }
          }
        })
      });
    });

    // Mock confusion matrix APIs for different models
    await page.route('**/api/confusion_matrix/*', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          confusion_matrix_image: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          confusion_matrix_normalized_image: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
          top_confused_pairs: []
        })
      });
    });

    await page.route('**/api/validation/*', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          report: {
            per_class_metrics: [],
            best_classes: [],
            worst_classes: []
          }
        })
      });
    });

    await page.goto('/frontend/stats.html');
    await page.waitForTimeout(2000);

    // Change model selection
    const confusionModelSelect = page.locator('#confusionModelSelect');
    await confusionModelSelect.selectOption('hybrid');

    await page.waitForTimeout(1000);

    // Check that confusion matrix container shows loading
    const confusionMatrixContainer = page.locator('#confusionMatrixContainer');
    await expect(confusionMatrixContainer).toBeVisible();
  });

  test('should update total classes from API', async ({ page }) => {
    await page.route('**/api/stats', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          total_classes: 42,
          statistics: {}
        })
      });
    });

    await page.goto('/frontend/stats.html');
    await page.waitForTimeout(1000);

    const totalClasses = page.locator('#totalClasses');
    await expect(totalClasses).toContainText('42');
  });
});
