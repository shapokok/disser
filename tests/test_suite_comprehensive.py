#!/usr/bin/env python3
"""
Comprehensive QA Test Suite for Crop Disease Detection System
Tests all backend endpoints, frontend integration, and error handling
"""

import requests
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import base64
import io

# Optional imports
try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/numpy not available. Image creation tests will be limited.")

# Configuration
API_BASE = "http://localhost:5000"
TEST_IMAGES_DIR = "../data/test_images"
UPLOAD_DIR = "../data/uploads"
RESULTS_FILE = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Test results tracker
test_results = {
    "timestamp": datetime.now().isoformat(),
    "tests_run": 0,
    "tests_passed": 0,
    "tests_failed": 0,
    "failures": [],
    "warnings": [],
    "test_details": []
}


class TestReporter:
    """Helper class for consistent test reporting"""

    @staticmethod
    def print_header(title):
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)

    @staticmethod
    def print_test(test_name, status="RUNNING"):
        symbols = {
            "RUNNING": "⏳",
            "PASS": "✓",
            "FAIL": "✗",
            "WARN": "⚠"
        }
        colors = {
            "RUNNING": "\033[94m",  # Blue
            "PASS": "\033[92m",      # Green
            "FAIL": "\033[91m",      # Red
            "WARN": "\033[93m",      # Yellow
            "END": "\033[0m"         # Reset
        }

        symbol = symbols.get(status, "?")
        color = colors.get(status, "")
        end_color = colors["END"]

        print(f"{color}{symbol} {status:8}{end_color} | {test_name}")

    @staticmethod
    def record_result(test_name, passed, details="", warning=False):
        test_results["tests_run"] += 1

        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

        if passed:
            test_results["tests_passed"] += 1
            TestReporter.print_test(test_name, "PASS")
        else:
            test_results["tests_failed"] += 1
            test_results["failures"].append({
                "test": test_name,
                "details": details
            })
            TestReporter.print_test(test_name, "FAIL")

        if warning:
            test_results["warnings"].append({
                "test": test_name,
                "details": details
            })
            TestReporter.print_test(f"  └─ Warning: {details}", "WARN")

        test_results["test_details"].append(result)
        return passed


def create_test_image(filename="test_image.jpg", size=(224, 224)):
    """Create a test image for upload testing"""
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

    filepath = os.path.join(TEST_IMAGES_DIR, filename)

    if HAS_PIL:
        # Create a simple test image with random colors
        img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(filepath, 'JPEG')
    else:
        # Create a minimal valid JPEG file (1x1 pixel)
        # JPEG header for a minimal 1x1 red pixel image
        jpeg_data = bytes.fromhex(
            'ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707'
            '07090908080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c'
            '231c1c2837292c30313434341f27393d38323c2e333432ffdb0043010909090c0b'
            '0c180d0d1832211c2132323232323232323232323232323232323232323232323232'
            '323232323232323232323232323232323232323232323232323232ffc00011080001'
            '000103012200021101031101ffc4001500010100000000000000000000000000000008'
            'ffc40014100100000000000000000000000000000000ffc4001501010100000000000000'
            '000000000000000008ffc4001411010000000000000000000000000000000000ffda'
            '000c03010002110311003f00bf800000ffd9'
        )
        with open(filepath, 'wb') as f:
            f.write(jpeg_data)

    return filepath


def test_1_environment_check():
    """Test 1: Environment and server availability"""
    TestReporter.print_header("TEST 1: Environment Check")

    # Test 1.1: Backend server running
    TestReporter.print_test("Backend server connectivity", "RUNNING")
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            TestReporter.record_result(
                "Backend server running",
                True,
                f"Server status: {data.get('status')}, Version: {data.get('version')}"
            )
        else:
            TestReporter.record_result(
                "Backend server running",
                False,
                f"HTTP {response.status_code}"
            )
            return False
    except Exception as e:
        TestReporter.record_result(
            "Backend server running",
            False,
            f"Connection error: {str(e)}"
        )
        return False

    # Test 1.2: Models loaded
    TestReporter.print_test("Models loaded", "RUNNING")
    try:
        response = requests.get(f"{API_BASE}/api/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            model_names = [m['name'] for m in models]

            expected_models = ['baseline', 'efficientnet', 'mobilenet', 'hybrid']
            loaded_models = [m for m in expected_models if m in model_names]

            if len(loaded_models) > 0:
                TestReporter.record_result(
                    "Models loaded",
                    True,
                    f"Loaded: {', '.join(loaded_models)}",
                    warning=(len(loaded_models) < 4)
                )
            else:
                TestReporter.record_result(
                    "Models loaded",
                    False,
                    "No models available"
                )
        else:
            TestReporter.record_result("Models loaded", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("Models loaded", False, str(e))

    # Test 1.3: Upload directory accessible
    TestReporter.print_test("Upload directory accessible", "RUNNING")
    upload_path = Path(UPLOAD_DIR)
    if upload_path.exists() or upload_path.parent.exists():
        TestReporter.record_result("Upload directory accessible", True)
    else:
        TestReporter.record_result("Upload directory accessible", False, "Directory not found")

    return True


def test_2_image_upload():
    """Test 2: Image upload functionality"""
    TestReporter.print_header("TEST 2: Image Upload")

    # Test 2.1: Single image upload (JPEG)
    TestReporter.print_test("Upload single JPEG image", "RUNNING")
    test_image = create_test_image("test_upload.jpg")

    try:
        with open(test_image, 'rb') as f:
            files = {'files': ('test_upload.jpg', f, 'image/jpeg')}
            response = requests.post(f"{API_BASE}/api/upload", files=files)

        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('uploaded'):
                TestReporter.record_result(
                    "Upload single JPEG",
                    True,
                    f"Uploaded: {data['uploaded'][0]['original_name']}"
                )
                # Store uploaded file info for later tests
                global uploaded_file
                uploaded_file = data['uploaded'][0]
            else:
                TestReporter.record_result("Upload single JPEG", False, "No files in response")
        else:
            TestReporter.record_result("Upload single JPEG", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("Upload single JPEG", False, str(e))

    # Test 2.2: Multiple images upload
    TestReporter.print_test("Upload multiple images", "RUNNING")
    test_images = [
        create_test_image(f"test_multi_{i}.jpg") for i in range(3)
    ]

    try:
        files = [
            ('files', (os.path.basename(img), open(img, 'rb'), 'image/jpeg'))
            for img in test_images
        ]
        response = requests.post(f"{API_BASE}/api/upload", files=files)

        # Close file handles
        for _, (_, fh, _) in files:
            fh.close()

        if response.status_code == 200:
            data = response.json()
            count = len(data.get('uploaded', []))
            TestReporter.record_result(
                "Upload multiple images",
                count == 3,
                f"Uploaded {count}/3 files"
            )
        else:
            TestReporter.record_result("Upload multiple images", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("Upload multiple images", False, str(e))

    # Test 2.3: JFIF format support
    TestReporter.print_test("Upload JFIF image", "RUNNING")
    jfif_image = create_test_image("test_upload.jfif")

    try:
        with open(jfif_image, 'rb') as f:
            files = {'files': ('test_upload.jfif', f, 'image/jpeg')}
            response = requests.post(f"{API_BASE}/api/upload", files=files)

        if response.status_code == 200:
            data = response.json()
            TestReporter.record_result(
                "Upload JFIF format",
                data.get('success', False),
                "JFIF support verified"
            )
        else:
            TestReporter.record_result("Upload JFIF format", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("Upload JFIF format", False, str(e))

    # Test 2.4: Invalid file type rejection
    TestReporter.print_test("Reject invalid file types", "RUNNING")
    try:
        # Create a text file pretending to be an image
        invalid_file = os.path.join(TEST_IMAGES_DIR, "invalid.txt")
        with open(invalid_file, 'w') as f:
            f.write("This is not an image")

        with open(invalid_file, 'rb') as f:
            files = {'files': ('invalid.txt', f, 'text/plain')}
            response = requests.post(f"{API_BASE}/api/upload", files=files)

        data = response.json()
        # Should succeed but with errors for invalid files
        has_errors = len(data.get('errors', [])) > 0 or len(data.get('uploaded', [])) == 0

        TestReporter.record_result(
            "Reject invalid file types",
            has_errors,
            "Invalid file type properly rejected" if has_errors else "Invalid file was accepted!"
        )
    except Exception as e:
        TestReporter.record_result("Reject invalid file types", False, str(e))


def test_3_disease_classification():
    """Test 3: Disease classification with all models"""
    TestReporter.print_header("TEST 3: Disease Classification")

    # Get available models
    try:
        response = requests.get(f"{API_BASE}/api/models")
        models_data = response.json()
        available_models = [m['name'] for m in models_data.get('models', [])]
    except:
        available_models = ['baseline', 'efficientnet', 'mobilenet', 'hybrid']

    # Create and upload a test image
    test_image_path = create_test_image("test_classification.jpg")

    with open(test_image_path, 'rb') as f:
        files = {'files': ('test_classification.jpg', f, 'image/jpeg')}
        upload_response = requests.post(f"{API_BASE}/api/upload", files=files)

    if upload_response.status_code != 200:
        TestReporter.record_result("Classification - Upload", False, "Failed to upload test image")
        return

    uploaded_data = upload_response.json()
    image_saved_name = uploaded_data['uploaded'][0]['saved_name']

    # Test each model
    for model_name in available_models:
        TestReporter.print_test(f"Classification with {model_name}", "RUNNING")

        try:
            payload = {
                "image_path": image_saved_name,
                "model": model_name,
                "explanation": "gradcam",
                "dataset_type": "controlled"
            }

            response = requests.post(
                f"{API_BASE}/api/predict",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                data = response.json()

                # Check required fields
                has_prediction = 'prediction' in data
                has_class = data.get('prediction', {}).get('class') is not None
                has_confidence = 'confidence' in data.get('prediction', {})
                has_model = data.get('model_used') == model_name

                all_valid = has_prediction and has_class and has_confidence and has_model

                details = f"Class: {data.get('prediction', {}).get('class', 'N/A')}, " \
                         f"Confidence: {data.get('prediction', {}).get('confidence_percent', 'N/A')}"

                TestReporter.record_result(
                    f"Prediction with {model_name}",
                    all_valid,
                    details
                )
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                TestReporter.record_result(
                    f"Prediction with {model_name}",
                    False,
                    f"HTTP {response.status_code}: {error_data.get('error', 'Unknown error')}"
                )
        except Exception as e:
            TestReporter.record_result(f"Prediction with {model_name}", False, str(e))


def test_4_visualizations():
    """Test 4: Grad-CAM and LIME visualizations"""
    TestReporter.print_header("TEST 4: Visualizations (Grad-CAM & LIME)")

    # Create and upload test image
    test_image_path = create_test_image("test_viz.jpg")

    with open(test_image_path, 'rb') as f:
        files = {'files': ('test_viz.jpg', f, 'image/jpeg')}
        upload_response = requests.post(f"{API_BASE}/api/upload", files=files)

    if upload_response.status_code != 200:
        TestReporter.record_result("Visualization - Upload", False, "Failed to upload test image")
        return

    image_saved_name = upload_response.json()['uploaded'][0]['saved_name']

    # Test 4.1: Grad-CAM visualization
    TestReporter.print_test("Grad-CAM visualization", "RUNNING")
    try:
        payload = {
            "image_path": image_saved_name,
            "model": "efficientnet",
            "explanation": "gradcam",
            "dataset_type": "controlled"
        }

        response = requests.post(f"{API_BASE}/api/predict", json=payload)

        if response.status_code == 200:
            data = response.json()
            has_visualization = data.get('visualization') is not None

            # Verify it's valid base64 image
            if has_visualization:
                try:
                    img_data = base64.b64decode(data['visualization'])
                    if HAS_PIL:
                        img = Image.open(io.BytesIO(img_data))
                        is_valid = img.size[0] > 0 and img.size[1] > 0
                    else:
                        # Just check if it's valid base64 and has data
                        is_valid = len(img_data) > 100
                except:
                    is_valid = False
            else:
                is_valid = False

            TestReporter.record_result(
                "Grad-CAM visualization",
                is_valid,
                f"Visualization generated: {has_visualization}, Valid: {is_valid}"
            )
        else:
            TestReporter.record_result("Grad-CAM visualization", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("Grad-CAM visualization", False, str(e))

    # Test 4.2: LIME visualization
    TestReporter.print_test("LIME visualization", "RUNNING")
    try:
        payload = {
            "image_path": image_saved_name,
            "model": "efficientnet",
            "explanation": "lime",
            "dataset_type": "controlled"
        }

        response = requests.post(f"{API_BASE}/api/predict", json=payload)

        if response.status_code == 200:
            data = response.json()
            has_visualization = data.get('visualization') is not None

            TestReporter.record_result(
                "LIME visualization",
                has_visualization,
                "LIME explanation generated successfully"
            )
        else:
            TestReporter.record_result("LIME visualization", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("LIME visualization", False, str(e))


def test_5_model_comparison():
    """Test 5: Model comparison endpoint"""
    TestReporter.print_header("TEST 5: Model Comparison")

    # Create and upload test image
    test_image_path = create_test_image("test_compare.jpg")

    with open(test_image_path, 'rb') as f:
        files = {'files': ('test_compare.jpg', f, 'image/jpeg')}
        upload_response = requests.post(f"{API_BASE}/api/upload", files=files)

    if upload_response.status_code != 200:
        TestReporter.record_result("Model comparison - Upload", False, "Failed to upload")
        return

    image_saved_name = upload_response.json()['uploaded'][0]['saved_name']

    TestReporter.print_test("Compare all models", "RUNNING")
    try:
        payload = {
            "image_path": image_saved_name,
            "models": ["baseline", "efficientnet", "mobilenet", "hybrid"]
        }

        response = requests.post(f"{API_BASE}/api/compare", json=payload)

        if response.status_code == 200:
            data = response.json()
            comparisons = data.get('comparisons', {})

            # Count how many models returned results
            model_count = len(comparisons)
            has_results = model_count > 0

            # Check if each result has required fields
            all_valid = all(
                'predicted_class' in comp and 'confidence' in comp
                for comp in comparisons.values()
            )

            TestReporter.record_result(
                "Model comparison",
                has_results and all_valid,
                f"Compared {model_count} models successfully"
            )
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            TestReporter.record_result(
                "Model comparison",
                False,
                f"HTTP {response.status_code}: {error_data.get('error', 'Unknown')}"
            )
    except Exception as e:
        TestReporter.record_result("Model comparison", False, str(e))


def test_6_metrics_endpoints():
    """Test 6: Metrics and statistics endpoints"""
    TestReporter.print_header("TEST 6: Metrics & Statistics")

    # Test 6.1: General statistics
    TestReporter.print_test("Get model statistics", "RUNNING")
    try:
        response = requests.get(f"{API_BASE}/api/stats")

        if response.status_code == 200:
            data = response.json()
            has_stats = 'statistics' in data
            has_models = 'models_available' in data

            TestReporter.record_result(
                "Model statistics endpoint",
                has_stats and has_models,
                f"Models available: {data.get('models_available', [])}"
            )
        else:
            TestReporter.record_result("Model statistics endpoint", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("Model statistics endpoint", False, str(e))

    # Test 6.2: Class names
    TestReporter.print_test("Get class names", "RUNNING")
    try:
        response = requests.get(f"{API_BASE}/api/classes")

        if response.status_code == 200:
            data = response.json()
            classes = data.get('classes', [])
            has_classes = len(classes) > 0

            TestReporter.record_result(
                "Class names endpoint",
                has_classes,
                f"Total classes: {len(classes)}"
            )
        else:
            TestReporter.record_result("Class names endpoint", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("Class names endpoint", False, str(e))

    # Test 6.3: Confusion matrix
    TestReporter.print_test("Get confusion matrix", "RUNNING")
    try:
        response = requests.get(f"{API_BASE}/api/confusion_matrix/hybrid")

        if response.status_code == 200:
            data = response.json()
            has_matrix = 'confusion_matrix' in data
            has_image = 'confusion_matrix_image' in data

            TestReporter.record_result(
                "Confusion matrix endpoint",
                has_matrix and has_image,
                "Confusion matrix data available"
            )
        else:
            TestReporter.record_result("Confusion matrix endpoint", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("Confusion matrix endpoint", False, str(e))

    # Test 6.4: Validation report
    TestReporter.print_test("Get validation report", "RUNNING")
    try:
        response = requests.get(f"{API_BASE}/api/validation/hybrid")

        if response.status_code == 200:
            data = response.json()
            has_report = 'report' in data

            TestReporter.record_result(
                "Validation report endpoint",
                has_report,
                "Validation report available"
            )
        else:
            TestReporter.record_result("Validation report endpoint", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("Validation report endpoint", False, str(e))


def test_7_error_handling():
    """Test 7: Error handling"""
    TestReporter.print_header("TEST 7: Error Handling")

    # Test 7.1: Missing image path
    TestReporter.print_test("Handle missing image path", "RUNNING")
    try:
        payload = {
            "model": "efficientnet",
            "explanation": "gradcam"
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload)

        # Should return 400 or similar error
        is_error = response.status_code >= 400
        has_error_msg = 'error' in response.json() if is_error else False

        TestReporter.record_result(
            "Missing image path error handling",
            is_error and has_error_msg,
            f"Returned HTTP {response.status_code} with error message"
        )
    except Exception as e:
        TestReporter.record_result("Missing image path error handling", False, str(e))

    # Test 7.2: Invalid model name
    TestReporter.print_test("Handle invalid model name", "RUNNING")
    try:
        # First upload an image
        test_image_path = create_test_image("test_error.jpg")
        with open(test_image_path, 'rb') as f:
            files = {'files': ('test_error.jpg', f, 'image/jpeg')}
            upload_response = requests.post(f"{API_BASE}/api/upload", files=files)

        image_saved_name = upload_response.json()['uploaded'][0]['saved_name']

        payload = {
            "image_path": image_saved_name,
            "model": "nonexistent_model",
            "explanation": "gradcam"
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload)

        is_error = response.status_code >= 400
        has_error_msg = 'error' in response.json() if is_error else False

        TestReporter.record_result(
            "Invalid model name error handling",
            is_error and has_error_msg,
            f"Returned HTTP {response.status_code}"
        )
    except Exception as e:
        TestReporter.record_result("Invalid model name error handling", False, str(e))

    # Test 7.3: Nonexistent image path
    TestReporter.print_test("Handle nonexistent image", "RUNNING")
    try:
        payload = {
            "image_path": "nonexistent_image_12345.jpg",
            "model": "efficientnet",
            "explanation": "gradcam"
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload)

        is_error = response.status_code >= 400
        has_error_msg = 'error' in response.json() if is_error else False

        TestReporter.record_result(
            "Nonexistent image error handling",
            is_error and has_error_msg,
            f"Returned HTTP {response.status_code}"
        )
    except Exception as e:
        TestReporter.record_result("Nonexistent image error handling", False, str(e))

    # Test 7.4: Invalid endpoint
    TestReporter.print_test("Handle invalid endpoint", "RUNNING")
    try:
        response = requests.get(f"{API_BASE}/api/invalid_endpoint_xyz")

        is_error = response.status_code == 404

        TestReporter.record_result(
            "Invalid endpoint error handling",
            is_error,
            f"Returned HTTP {response.status_code}"
        )
    except Exception as e:
        TestReporter.record_result("Invalid endpoint error handling", False, str(e))


def test_8_batch_processing():
    """Test 8: Batch processing"""
    TestReporter.print_header("TEST 8: Batch Processing")

    # Create and upload multiple test images
    test_images = [create_test_image(f"batch_{i}.jpg") for i in range(3)]

    uploaded_names = []
    for img_path in test_images:
        with open(img_path, 'rb') as f:
            files = {'files': (os.path.basename(img_path), f, 'image/jpeg')}
            response = requests.post(f"{API_BASE}/api/upload", files=files)

        if response.status_code == 200:
            uploaded_names.append(response.json()['uploaded'][0]['saved_name'])

    if len(uploaded_names) < 3:
        TestReporter.record_result("Batch processing - Upload", False, "Failed to upload test images")
        return

    TestReporter.print_test("Batch image processing", "RUNNING")
    try:
        payload = {
            "image_paths": uploaded_names,
            "model": "efficientnet",
            "explanation": "gradcam"
        }

        response = requests.post(f"{API_BASE}/api/batch", json=payload)

        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])

            # Check if all images were processed
            all_processed = len(results) == 3
            all_successful = all(r.get('success', False) or 'predicted_class' in r for r in results)

            TestReporter.record_result(
                "Batch processing",
                all_processed and all_successful,
                f"Processed {len(results)}/3 images"
            )
        else:
            TestReporter.record_result("Batch processing", False, f"HTTP {response.status_code}")
    except Exception as e:
        TestReporter.record_result("Batch processing", False, str(e))


def generate_report():
    """Generate and save detailed test report"""
    TestReporter.print_header("TEST SUMMARY")

    total = test_results["tests_run"]
    passed = test_results["tests_passed"]
    failed = test_results["tests_failed"]
    pass_rate = (passed / total * 100) if total > 0 else 0

    print(f"\nTotal Tests Run:    {total}")
    print(f"Tests Passed:       {passed} (\033[92m{pass_rate:.1f}%\033[0m)")
    print(f"Tests Failed:       {failed}")
    print(f"Warnings:           {len(test_results['warnings'])}")

    if failed > 0:
        print(f"\n\033[91mFailed Tests:\033[0m")
        for failure in test_results['failures']:
            print(f"  ✗ {failure['test']}")
            print(f"    └─ {failure['details']}")

    if test_results['warnings']:
        print(f"\n\033[93mWarnings:\033[0m")
        for warning in test_results['warnings']:
            print(f"  ⚠ {warning['test']}")
            print(f"    └─ {warning['details']}")

    # Save to JSON file
    with open(RESULTS_FILE, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n\033[94mDetailed results saved to: {RESULTS_FILE}\033[0m")

    return pass_rate >= 80


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  CROP DISEASE DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  API Endpoint: {API_BASE}")
    print("="*80)

    try:
        # Run all test suites
        if not test_1_environment_check():
            print("\n\033[91m⚠ Environment check failed. Cannot continue testing.\033[0m")
            sys.exit(1)

        test_2_image_upload()
        test_3_disease_classification()
        test_4_visualizations()
        test_5_model_comparison()
        test_6_metrics_endpoints()
        test_7_error_handling()
        test_8_batch_processing()

        # Generate report
        success = generate_report()

        print("\n" + "="*80)
        if success:
            print("  \033[92m✓ TEST SUITE COMPLETED SUCCESSFULLY\033[0m")
        else:
            print("  \033[91m✗ TEST SUITE COMPLETED WITH FAILURES\033[0m")
        print("="*80 + "\n")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n\033[93mTest suite interrupted by user\033[0m")
        generate_report()
        sys.exit(1)
    except Exception as e:
        print(f"\n\033[91mFatal error: {str(e)}\033[0m")
        generate_report()
        sys.exit(1)


if __name__ == "__main__":
    main()
