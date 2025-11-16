# Testing Documentation - Crop Disease Detection System

## Overview

This project includes comprehensive automated testing using **Playwright** for end-to-end browser testing of the frontend application.

## ✅ What's Covered

### Frontend UI Tests (71 test cases)
- **Home Page (18 tests)** - Landing page, navigation, features, models
- **Analyze Page (31 tests)** - Image upload, model selection, API integration
- **Statistics Page (22 tests)** - Performance metrics, charts, confusion matrices

### Cross-Browser Testing
- ✅ Chromium (Chrome/Edge)
- ✅ Firefox
- ✅ WebKit (Safari)
- ✅ Mobile Chrome (Pixel 5)
- ✅ Mobile Safari (iPhone 12)

### CI/CD Integration
- ✅ Automated testing on GitHub Actions
- ✅ Test reports and artifacts
- ✅ Parallel execution across browsers
- ✅ Automatic retries on failure

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Playwright browsers
npx playwright install
```

### 2. Run Tests

```bash
# Run all tests
npm test

# Run with UI (interactive)
npm run test:ui

# Run in headed mode (see browser)
npm run test:headed

# View test report
npm run test:report
```

## 📁 Project Structure

```
disser/
├── tests/                          # Playwright test files
│   ├── home.spec.js               # Home page tests
│   ├── analyze.spec.js            # Analyze page tests
│   ├── stats.spec.js              # Statistics page tests
│   └── README.md                  # Detailed test documentation
├── .github/
│   └── workflows/
│       └── playwright-tests.yml   # CI/CD workflow
├── playwright.config.js           # Playwright configuration
├── package.json                   # Node dependencies
├── .gitignore                     # Ignored files
└── TESTING.md                     # This file
```

## 🧪 Test Execution

### Local Development

```bash
# Run all tests (headless)
npm test

# Run specific browser
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit

# Run specific test file
npx playwright test tests/home.spec.js

# Debug specific test
npx playwright test --debug tests/analyze.spec.js

# Generate code (for writing new tests)
npm run test:codegen
```

### CI/CD (GitHub Actions)

Tests run automatically on:
- Every push to `main`, `master`, `develop`, or `claude/*` branches
- Every pull request to `main`, `master`, or `develop`
- Manual trigger via "Actions" tab

**View Results:**
1. Go to GitHub repository → **Actions** tab
2. Click on latest workflow run
3. View test results and download artifacts

## 📊 Test Coverage

| Category | Test Count | Description |
|----------|------------|-------------|
| Navigation | 12 | Header, links, logo, mobile menu |
| Content Display | 20 | Text, images, cards, sections |
| User Interactions | 15 | Buttons, forms, file uploads |
| API Integration | 10 | Mocked API calls, error handling |
| Responsive Design | 8 | Mobile viewports, breakpoints |
| Accessibility | 6 | Keyboard navigation, meta tags |
| **Total** | **71** | **Comprehensive Coverage** |

## 🔧 Configuration

### Backend Server

Tests require the Flask backend running on `http://localhost:5000`.

**Auto-start (default):**
- Playwright config includes `webServer` option
- Backend starts automatically before tests
- Stops automatically after tests complete

**Manual start:**
```bash
cd crop_monitoring_app
python backend/app.py
```

### Environment Variables

```bash
# Change base URL
BASE_URL=http://localhost:5001 npm test

# Run in CI mode (more retries)
CI=true npm test
```

## 📈 Reports

### HTML Report (Interactive)

```bash
# Generate and view
npm run test:report
```

Features:
- Test results by browser
- Screenshots on failure
- Video recordings
- Trace viewer
- Detailed error messages

### JUnit XML Report

Generated automatically in `test-results/junit.xml` for CI integration.

## 🐛 Debugging

### Debug Single Test

```bash
# Open Playwright Inspector
npx playwright test --debug tests/home.spec.js:10
```

### View Trace

```bash
# If test failed and trace was collected
npx playwright show-trace test-results/trace.zip
```

### Common Issues

**Backend not running:**
```bash
cd crop_monitoring_app
python backend/app.py
```

**Port 5000 in use:**
```bash
# Kill process
lsof -ti:5000 | xargs kill -9
```

**Browsers not installed:**
```bash
npx playwright install --with-deps
```

## 📝 Writing Tests

### Test Template

```javascript
const { test, expect } = require('@playwright/test');

test.describe('Feature Name', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/your-page.html');
  });

  test('should perform action', async ({ page }) => {
    // Arrange
    const button = page.locator('#submit-btn');

    // Act
    await button.click();

    // Assert
    await expect(button).toHaveClass('active');
  });
});
```

### Best Practices

1. ✅ Use descriptive test names
2. ✅ Keep tests independent
3. ✅ Mock API responses
4. ✅ Use proper selectors (prefer `data-testid`)
5. ✅ Test both success and error cases
6. ✅ Avoid hard-coded waits
7. ✅ Clean up after tests

## 🚦 CI/CD Pipeline

### Workflow Steps

1. **Checkout** - Clone repository
2. **Setup** - Install Node.js 18 and Python 3.9
3. **Install** - Install dependencies (npm + pip)
4. **Browsers** - Install Playwright browsers
5. **Backend** - Start Flask server
6. **Test** - Run Playwright tests (parallel)
7. **Upload** - Store test reports and screenshots

### Test Matrix

| Job | Browsers | Duration |
|-----|----------|----------|
| Desktop Tests | Chromium, Firefox, WebKit | ~3-5 min |
| Mobile Tests | Chrome Mobile, Safari Mobile | ~1-2 min |

### Artifacts

- **Test Reports** (HTML) - 30 days retention
- **Screenshots** (on failure) - 30 days retention
- **JUnit XML** - For test result tracking

## 📚 Additional Resources

- **Playwright Docs:** https://playwright.dev
- **Test Documentation:** `tests/README.md`
- **GitHub Actions:** `.github/workflows/playwright-tests.yml`
- **Configuration:** `playwright.config.js`

## 🤝 Contributing

When adding new features:

1. Write tests for new UI components
2. Run tests locally: `npm test`
3. Ensure all tests pass
4. Update documentation if needed
5. Commit and push (CI will run tests)

## 📞 Support

For testing issues:

1. Check `tests/README.md` for detailed documentation
2. Review Playwright docs: https://playwright.dev
3. Check CI logs on GitHub Actions
4. Run tests with `--debug` flag for troubleshooting

---

**Testing Stack:**
- **Framework:** Playwright 1.40+
- **Language:** JavaScript/Node.js
- **Browsers:** Chromium, Firefox, WebKit
- **CI/CD:** GitHub Actions
- **Reports:** HTML, JUnit XML

**Status:** ✅ All 71 tests passing across 5 browser configurations
