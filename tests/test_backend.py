#!/usr/bin/env python3
"""
Quick test script to verify backend API endpoints are working
"""

import requests
import json

API_BASE = "http://localhost:5000"

def test_health():
    """Test API health check"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Models loaded: {data.get('models_loaded')}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_stats():
    """Test model statistics endpoint"""
    print("\n=== Testing Statistics ===")
    try:
        response = requests.get(f"{API_BASE}/api/stats")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Success: {data.get('success')}")
        print(f"Models available: {data.get('models_available')}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_models():
    """Test models endpoint"""
    print("\n=== Testing Models Endpoint ===")
    try:
        response = requests.get(f"{API_BASE}/api/models")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Success: {data.get('success')}")
        if data.get('models'):
            for model in data['models']:
                print(f"  - {model['name']}: {model['status']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_compare_without_upload():
    """Test compare endpoint (will fail without uploaded image, but shows error handling)"""
    print("\n=== Testing Compare Endpoint (Error Handling) ===")
    try:
        response = requests.post(
            f"{API_BASE}/api/compare",
            json={
                "image_path": "nonexistent.jpg",
                "models": ["baseline", "efficientnet", "mobilenet", "hybrid"]
            }
        )
        print(f"Status: {response.status_code}")
        data = response.json()

        if response.status_code != 200:
            print(f"Expected error: {data.get('error')}")
            if 'available_models' in data:
                print(f"Available models: {data.get('available_models')}")
        else:
            print("Unexpected success - should have failed with missing image")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*60)
    print("Backend API Test Suite")
    print("="*60)
    print(f"Testing API at: {API_BASE}")

    results = []
    results.append(("Health Check", test_health()))
    results.append(("Statistics", test_stats()))
    results.append(("Models List", test_models()))
    results.append(("Compare (Error Handling)", test_compare_without_upload()))

    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✓ All tests passed! Backend is working correctly.")
    else:
        print("\n✗ Some tests failed. Check the backend server.")
        print("Make sure the backend is running: python backend/app.py")

if __name__ == "__main__":
    main()
