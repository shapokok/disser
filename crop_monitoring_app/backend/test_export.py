"""
Test script for export_utils module
Run this inside the container to verify Excel export functionality
"""

from export_utils import (
    export_to_excel,
    create_analysis_excel,
    create_batch_excel,
    create_comparison_excel,
    create_validation_excel
)
import json


def test_analysis_export():
    """Test analysis report export"""
    print("\n" + "="*50)
    print("Testing Analysis Export...")
    print("="*50)

    sample_data = {
        'image_name': 'tomato_leaf_sample.jpg',
        'model_used': 'efficientnet',
        'explanation_method': 'gradcam',
        'dataset_type': 'field',
        'inference_time_ms': 145.23,
        'prediction': {
            'class': 'Tomato - Early Blight',
            'class_raw': 'Tomato___Early_Blight',
            'confidence': 0.92,
            'confidence_percent': '92.00%'
        },
        'top_predictions': [
            {'class': 'Tomato - Early Blight', 'confidence': 0.92, 'confidence_percent': '92.00%'},
            {'class': 'Tomato - Septoria Leaf Spot', 'confidence': 0.05, 'confidence_percent': '5.00%'},
            {'class': 'Tomato - Healthy', 'confidence': 0.02, 'confidence_percent': '2.00%'}
        ]
    }

    try:
        excel_bytes = create_analysis_excel(sample_data)
        print(f"✓ Analysis export successful! Generated {len(excel_bytes)} bytes")
        return True
    except Exception as e:
        print(f"✗ Analysis export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_export():
    """Test batch processing export"""
    print("\n" + "="*50)
    print("Testing Batch Export...")
    print("="*50)

    sample_data = {
        'total': 3,
        'results': [
            {
                'image_path': '/uploads/image1.jpg',
                'predicted_class': 'Corn - Common Rust',
                'confidence': 0.87
            },
            {
                'image_path': '/uploads/image2.jpg',
                'predicted_class': 'Potato - Late Blight',
                'confidence': 0.94
            },
            {
                'image_path': '/uploads/image3.jpg',
                'error': 'Image not found'
            }
        ]
    }

    try:
        excel_bytes = create_batch_excel(sample_data)
        print(f"✓ Batch export successful! Generated {len(excel_bytes)} bytes")
        return True
    except Exception as e:
        print(f"✗ Batch export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_export():
    """Test model comparison export"""
    print("\n" + "="*50)
    print("Testing Comparison Export...")
    print("="*50)

    sample_data = {
        'image_path': '/uploads/test_image.jpg',
        'comparisons': {
            'baseline': {
                'predicted_class': 'Tomato - Early Blight',
                'confidence_percent': '78.50%',
                'inference_time_ms': 89.23
            },
            'efficientnet': {
                'predicted_class': 'Tomato - Early Blight',
                'confidence_percent': '92.30%',
                'inference_time_ms': 145.67
            },
            'mobilenet': {
                'predicted_class': 'Tomato - Septoria Leaf Spot',
                'confidence_percent': '65.40%',
                'inference_time_ms': 67.89
            },
            'hybrid': {
                'predicted_class': 'Tomato - Early Blight',
                'confidence_percent': '94.20%',
                'inference_time_ms': 234.12
            }
        }
    }

    try:
        excel_bytes = create_comparison_excel(sample_data)
        print(f"✓ Comparison export successful! Generated {len(excel_bytes)} bytes")
        return True
    except Exception as e:
        print(f"✗ Comparison export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_export():
    """Test validation report export"""
    print("\n" + "="*50)
    print("Testing Validation Export...")
    print("="*50)

    sample_data = {
        'overall_metrics': {
            'accuracy': 0.9245,
            'precision': 0.9123,
            'recall': 0.9087,
            'f1_score': 0.9105
        },
        'per_class_metrics': [
            {
                'class_name': 'Tomato - Early Blight',
                'precision': 0.93,
                'recall': 0.91,
                'f1_score': 0.92,
                'support': 150
            },
            {
                'class_name': 'Tomato - Healthy',
                'precision': 0.98,
                'recall': 0.96,
                'f1_score': 0.97,
                'support': 200
            },
            {
                'class_name': 'Corn - Common Rust',
                'precision': 0.89,
                'recall': 0.87,
                'f1_score': 0.88,
                'support': 120
            }
        ]
    }

    try:
        excel_bytes = create_validation_excel(sample_data)
        print(f"✓ Validation export successful! Generated {len(excel_bytes)} bytes")
        return True
    except Exception as e:
        print(f"✗ Validation export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "🔬 "*25)
    print("Excel Export Utilities Test Suite")
    print("🔬 "*25)

    results = []

    # Run all tests
    results.append(('Analysis Export', test_analysis_export()))
    results.append(('Batch Export', test_batch_export()))
    results.append(('Comparison Export', test_comparison_export()))
    results.append(('Validation Export', test_validation_export()))

    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print("="*50)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Excel export is ready!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")

    return passed == total


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
