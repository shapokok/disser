#!/usr/bin/env python3
"""
Simple test script to verify export functionality works
"""

import sys
import os

# Add backend to path
sys.path.insert(0, '/home/user/disser/crop_monitoring_app/backend')

try:
    from export_utils import export_to_csv, export_to_json, export_to_excel
    print("✓ Export utilities imported successfully")
except ImportError as e:
    print(f"✗ Failed to import export utilities: {e}")
    sys.exit(1)

# Test data
test_results = [
    {
        'timestamp': '2025-11-15 23:30:00',
        'image_name': 'test_image_1.jpg',
        'predicted_class': 'Apple Scab',
        'confidence': 0.95,
        'model_used': 'EfficientNet',
        'explanation_method': 'Grad-CAM',
        'inference_time': 150,
        'top_predictions': [
            {'class': 'Apple Scab', 'confidence': 0.95},
            {'class': 'Healthy', 'confidence': 0.03},
            {'class': 'Rust', 'confidence': 0.02}
        ]
    },
    {
        'timestamp': '2025-11-15 23:30:05',
        'image_name': 'test_image_2.jpg',
        'predicted_class': 'Healthy',
        'confidence': 0.98,
        'model_used': 'MobileNet',
        'explanation_method': 'LIME',
        'inference_time': 120,
        'top_predictions': [
            {'class': 'Healthy', 'confidence': 0.98},
            {'class': 'Apple Scab', 'confidence': 0.01},
            {'class': 'Rust', 'confidence': 0.01}
        ]
    }
]

print("\nTesting export functions...\n")

# Test CSV export
try:
    csv_data = export_to_csv(test_results)
    print(f"✓ CSV export works ({len(csv_data)} bytes)")
    print(f"  Preview: {csv_data[:200]}...")
except Exception as e:
    print(f"✗ CSV export failed: {e}")

# Test JSON export
try:
    json_data = export_to_json(test_results)
    print(f"\n✓ JSON export works ({len(json_data)} bytes)")
    print(f"  Preview: {json_data[:200]}...")
except Exception as e:
    print(f"\n✗ JSON export failed: {e}")

# Test Excel export
try:
    excel_data = export_to_excel(test_results)
    print(f"\n✓ Excel export works ({len(excel_data)} bytes)")
    print(f"  Generated Excel file with {len(test_results)} rows")
except Exception as e:
    print(f"\n✗ Excel export failed: {e}")

print("\n" + "="*50)
print("Export functionality test complete!")
print("="*50)
