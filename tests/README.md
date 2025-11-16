# Playwright E2E Tests - Crop Disease Detection System

Comprehensive end-to-end testing suite for the Crop Disease Detection web application using Playwright.

## Overview

This test suite provides complete coverage of the frontend UI, including:
- **Home Page** - Landing page with features and model descriptions
- **Analyze Page** - Image upload and disease detection interface
- **Statistics Page** - Model performance metrics and comparisons

## Test Structure

```
tests/
├── home.spec.js       # Home page tests (18 test cases)
├── analyze.spec.js    # Analyze page tests (31 test cases)
├── stats.spec.js      # Statistics page tests (22 test cases)
└── README.md         # This file
```

## Test Coverage

### Home Page Tests (18 tests)
✅ Page loading and title verification
✅ Navigation bar and logo
✅ Hero section with CTA buttons
✅ About section content
✅ 6 key features display
✅ Supported crops and diseases (8 crop types)
✅ 4 model architectures with stats
✅ Footer content
✅ Navigation to other pages
✅ Mobile menu toggle
✅ Responsive design
✅ Meta tags and CSS loading
✅ Accessibility features

### Analyze Page Tests (31 tests)
✅ Page loading and navigation
✅ Model selection (4 models)
✅ Explanation method selector (Grad-CAM, LIME)
✅ Dataset type selector (Controlled, Field)
✅ File upload area (drag & drop + click)
✅ File validation (type, size)
✅ Multiple file upload support
✅ Image preview functionality
✅ Analyze and Compare buttons
✅ Clear button functionality
✅ Loading states and progress indicators
✅ Results container
✅ API error handling (mocked)
✅ Responsive design

### Statistics Page Tests (22 tests)
✅ Page loading and navigation
✅ Loading indicator
✅ Dataset information (38 classes, 14 plant types, 54,306 images)
✅ Model recommendations (Best Accuracy, Speed, Balance)
✅ Training configuration details
✅ Confusion matrix selector
✅ Performance table display (mocked)
✅ Model cards display (mocked)
✅ Visual charts (accuracy, speed, F1-score)
✅ API integration with mocked responses
✅ Error handling
✅ Dynamic confusion matrix loading

## Prerequisites

### System Requirements
- Node.js 18+ (for Playwright)
- Python 3.9+ (for backend server)
- 2GB+ available RAM
- Internet connection (for first-time browser downloads)

### Dependencies

**Node Dependencies:**
```bash
npm install
```

**Python Dependencies:**
```bash
pip install flask pillow numpy
# Or install from requirements.txt if available
pip install -r crop_monitoring_app/requirements.txt
```

## Installation

1. **Install Node dependencies:**
   ```bash
   npm install
   ```

2. **Install Playwright browsers:**
   ```bash
   npx playwright install
   ```

3. **Install system dependencies (Linux only):**
   ```bash
   npx playwright install-deps
   ```

## Running Tests

### Run All Tests
```bash
npm test
```

### Run Specific Browser
```bash
# Chromium only
npx playwright test --project=chromium

# Firefox only
npx playwright test --project=firefox

# WebKit (Safari) only
npx playwright test --project=webkit
```

### Run Specific Test File
```bash
# Home page tests only
npx playwright test tests/home.spec.js

# Analyze page tests only
npx playwright test tests/analyze.spec.js

# Statistics page tests only
npx playwright test tests/stats.spec.js
```

### Run Tests in Headed Mode (See Browser)
```bash
npm run test:headed
```

### Run Tests with UI Mode (Interactive)
```bash
npm run test:ui
```

### Debug Tests
```bash
npm run test:debug
```

### View Test Report
```bash
npm run test:report
```

## Test Configuration

The test suite is configured in `playwright.config.js`:

- **Browsers:** Chromium, Firefox, WebKit
- **Mobile Devices:** Pixel 5, iPhone 12
- **Base URL:** `http://localhost:5000` (configurable via `BASE_URL` env var)
- **Timeouts:** 60s per test, 15s per action
- **Retries:** 2 retries in CI, 0 locally
- **Reporters:** HTML, List, JUnit
- **Screenshots:** On failure only
- **Video:** Retained on failure
- **Trace:** On first retry

## CI/CD Integration

The test suite runs automatically on GitHub Actions:

### Workflow Triggers
- Push to `main`, `master`, `develop`, or `claude/*` branches
- Pull requests to `main`, `master`, or `develop`
- Manual workflow dispatch

### Test Matrix
- **Desktop Browsers:** Chromium, Firefox, WebKit (parallel)
- **Mobile Viewports:** Chrome Mobile, Safari Mobile

### Artifacts
- Test reports (30-day retention)
- Screenshots on failure (30-day retention)
- JUnit XML reports

### Viewing CI Results
1. Go to the **Actions** tab in GitHub
2. Click on the latest workflow run
3. Download artifacts to view reports

## Backend Server

Tests require the Flask backend to be running on `http://localhost:5000`.

### Starting Backend Manually
```bash
cd crop_monitoring_app
python backend/app.py
```

### Auto-Start in Tests
The Playwright config includes a `webServer` option that automatically starts the backend before tests and stops it after.

## Test Data

Tests use mocked data for:
- API responses (statistics, predictions)
- Image uploads (base64-encoded 1x1 PNG)
- Confusion matrices
- Model comparisons

This ensures:
✅ **Fast execution** - No real ML model inference
✅ **Reliability** - Consistent, predictable results
✅ **Isolation** - Tests don't depend on backend state

## Writing New Tests

### Test Template
```javascript
const { test, expect } = require('@playwright/test');

test.describe('Feature Name', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/your-page.html');
  });

  test('should do something', async ({ page }) => {
    // Arrange
    const element = page.locator('#some-id');

    // Act
    await element.click();

    // Assert
    await expect(element).toHaveClass('active');
  });
});
```

### Best Practices
1. ✅ Use descriptive test names
2. ✅ Follow AAA pattern (Arrange, Act, Assert)
3. ✅ Use `data-testid` for stable selectors when possible
4. ✅ Mock external API calls
5. ✅ Test both happy path and error cases
6. ✅ Keep tests independent and isolated
7. ✅ Use `beforeEach` for common setup
8. ✅ Clean up after tests if needed

## Troubleshooting

### Tests Failing Locally

**Backend not running:**
```bash
# Start backend manually
cd crop_monitoring_app
python backend/app.py
```

**Port 5000 in use:**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or change BASE_URL
BASE_URL=http://localhost:5001 npm test
```

**Browsers not installed:**
```bash
npx playwright install
```

### Tests Timing Out

Increase timeout in `playwright.config.js`:
```javascript
timeout: 90 * 1000, // 90 seconds instead of 60
```

### Flaky Tests

Run with retries:
```bash
npx playwright test --retries=2
```

Or debug specific test:
```bash
npx playwright test --debug tests/home.spec.js:10
```

## Performance

### Test Execution Time
- **Full suite (all browsers):** ~3-5 minutes
- **Single browser:** ~1-2 minutes
- **Single test file:** ~20-40 seconds

### Optimization Tips
1. Run tests in parallel (default)
2. Use `fullyParallel: true` in config
3. Skip unnecessary waits
4. Mock API responses
5. Reuse browser contexts when possible

## Coverage Report

| Page | Tests | Coverage |
|------|-------|----------|
| Home | 18 | Navigation, Content, Responsive, Accessibility |
| Analyze | 31 | Upload, Selection, Validation, API, Loading |
| Statistics | 22 | Data Display, Charts, API, Error Handling |
| **Total** | **71** | **Comprehensive UI & Integration** |

## Maintenance

### Updating Tests

When UI changes:
1. Update selectors in affected tests
2. Run tests to verify
3. Update snapshots if needed
4. Commit changes

### Adding New Tests

1. Create test file in `tests/` directory
2. Follow naming convention: `feature.spec.js`
3. Add to CI workflow if needed
4. Update this README

## Resources

- [Playwright Documentation](https://playwright.dev)
- [Playwright Best Practices](https://playwright.dev/docs/best-practices)
- [GitHub Actions for Playwright](https://playwright.dev/docs/ci-intro)
- [Debugging Tests](https://playwright.dev/docs/debug)

## Support

For issues or questions:
1. Check [Playwright Docs](https://playwright.dev)
2. Review test logs and screenshots
3. Run tests with `--debug` flag
4. Check GitHub Actions logs for CI failures

## License

This test suite is part of the Crop Disease Detection System - Master's Thesis Project.
© 2025 - For Educational and Research Purposes
