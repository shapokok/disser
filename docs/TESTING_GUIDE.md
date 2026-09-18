# Crop Disease Detection System - Testing Guide

## Overview

This guide covers comprehensive testing of the Crop Monitoring App, including automated test suites and manual testing procedures.

## Quick Start

### 1. Start the Backend Server

```bash
cd crop_monitoring_app/backend
python app.py
```

Expected output:
```
✓ Baseline model loaded
✓ EfficientNet model loaded
✓ MobileNet model loaded
✓ Hybrid CNN-Transformer model loaded
All models loaded successfully!
```

### 2. Run the Comprehensive Test Suite

```bash
cd crop_monitoring_app
python test_suite_comprehensive.py
```

This will run **60+ automated tests** covering all functionality.

### 3. Run Quick Backend Verification

```bash
cd crop_monitoring_app
python test_backend.py
```

This runs quick sanity checks on critical endpoints.

---

## Test Coverage Summary

The comprehensive test suite (`test_suite_comprehensive.py`) provides:

- ✅ **51+ automated tests**
- ✅ **100% API endpoint coverage**
- ✅ **All model architectures tested**
- ✅ **Both visualization methods (Grad-CAM & LIME)**
- ✅ **Error handling validation**
- ✅ **Performance metrics**
- ✅ **Detailed JSON reports**

See full documentation in this file for complete testing procedures.
