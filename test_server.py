"""
Lightweight test server for Playwright E2E tests
Serves frontend files and provides mocked API responses
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import base64
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Get the directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(SCRIPT_DIR, 'crop_monitoring_app', 'frontend')

# Mock data
MOCK_STATS = {
    'baseline': {
        'accuracy': 0.923,
        'precision': 0.925,
        'recall': 0.921,
        'f1_score': 0.923,
        'inference_time_ms': 45,
        'parameters': '1.2M',
        'size_mb': '4.8'
    },
    'efficientnet': {
        'accuracy': 0.954,
        'precision': 0.956,
        'recall': 0.953,
        'f1_score': 0.954,
        'inference_time_ms': 58,
        'parameters': '4.0M',
        'size_mb': '16.2'
    },
    'mobilenet': {
        'accuracy': 0.941,
        'precision': 0.943,
        'recall': 0.939,
        'f1_score': 0.941,
        'inference_time_ms': 32,
        'parameters': '2.3M',
        'size_mb': '9.1'
    },
    'hybrid': {
        'accuracy': 0.967,
        'precision': 0.968,
        'recall': 0.966,
        'f1_score': 0.967,
        'inference_time_ms': 125,
        'parameters': '25.6M',
        'size_mb': '102.4'
    }
}

# 1x1 transparent PNG for testing
MOCK_IMAGE = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

@app.route('/')
def index():
    """Redirect to frontend index"""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/frontend/<path:path>')
def serve_frontend(path):
    """Serve frontend files"""
    return send_from_directory(FRONTEND_DIR, path)

@app.route('/style.css')
def serve_css():
    """Serve CSS file"""
    return send_from_directory(FRONTEND_DIR, 'style.css')

@app.route('/script.js')
def serve_js():
    """Serve JS file"""
    return send_from_directory(FRONTEND_DIR, 'script.js')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Mock statistics endpoint"""
    return jsonify({
        'success': True,
        'total_classes': 38,
        'statistics': MOCK_STATS
    })

@app.route('/api/upload', methods=['POST'])
def upload():
    """Mock upload endpoint"""
    files = request.files.getlist('files')
    uploaded = []

    for i, file in enumerate(files):
        uploaded.append({
            'original_name': file.filename,
            'saved_name': f'test_{i}_{file.filename}'
        })

    return jsonify({
        'success': True,
        'uploaded': uploaded
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Mock prediction endpoint"""
    data = request.json

    return jsonify({
        'success': True,
        'model_used': data.get('model', 'efficientnet'),
        'explanation_method': data.get('explanation', 'gradcam'),
        'dataset_type': data.get('dataset_type', 'controlled'),
        'prediction': {
            'class': 'Tomato___healthy',
            'confidence': 0.95,
            'confidence_percent': '95.0%'
        },
        'top_predictions': [
            {'class': 'Tomato___healthy', 'confidence': 0.95, 'confidence_percent': '95.0%'},
            {'class': 'Tomato___Late_Blight', 'confidence': 0.03, 'confidence_percent': '3.0%'},
            {'class': 'Tomato___Early_Blight', 'confidence': 0.02, 'confidence_percent': '2.0%'}
        ],
        'visualization': MOCK_IMAGE,
        'inference_time_ms': 58
    })

@app.route('/api/compare', methods=['POST'])
def compare():
    """Mock comparison endpoint"""
    data = request.json
    models = data.get('models', ['baseline', 'efficientnet', 'mobilenet', 'hybrid'])

    comparisons = {}
    for model in models:
        comparisons[model] = {
            'predicted_class': 'Tomato___healthy',
            'confidence': 0.92 + (0.01 * models.index(model)),
            'confidence_percent': f'{(92 + models.index(model))}%',
            'inference_time_ms': MOCK_STATS.get(model, {}).get('inference_time_ms', 50)
        }

    return jsonify({
        'success': True,
        'comparisons': comparisons
    })

@app.route('/api/confusion_matrix/<model_name>', methods=['GET'])
def confusion_matrix(model_name):
    """Mock confusion matrix endpoint"""
    return jsonify({
        'success': True,
        'model_name': model_name,
        'confusion_matrix_image': MOCK_IMAGE,
        'confusion_matrix_normalized_image': MOCK_IMAGE,
        'top_confused_pairs': [
            {
                'true_class': 'Tomato___Early_Blight',
                'predicted_class': 'Tomato___Late_Blight',
                'count': 15,
                'percentage': 3.2
            }
        ]
    })

@app.route('/api/validation/<model_name>', methods=['GET'])
def validation(model_name):
    """Mock validation endpoint"""
    return jsonify({
        'success': True,
        'report': {
            'per_class_metrics': [
                {
                    'class_name': 'Tomato___healthy',
                    'precision': 0.99,
                    'recall': 0.98,
                    'f1_score': 0.985,
                    'support': 100
                },
                {
                    'class_name': 'Tomato___Early_Blight',
                    'precision': 0.94,
                    'recall': 0.92,
                    'f1_score': 0.93,
                    'support': 95
                }
            ],
            'best_classes': [
                {
                    'class_name': 'Tomato___healthy',
                    'precision': 0.99,
                    'recall': 0.98,
                    'f1_score': 0.985
                }
            ],
            'worst_classes': [
                {
                    'class_name': 'Tomato___Early_Blight',
                    'precision': 0.85,
                    'recall': 0.82,
                    'f1_score': 0.835
                }
            ]
        }
    })

if __name__ == '__main__':
    print("="*60)
    print("Playwright Test Server for Crop Disease Detection System")
    print("="*60)
    print(f"Frontend directory: {FRONTEND_DIR}")
    print(f"Server running on: http://localhost:5000")
    print("="*60)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Disable debug in CI
        threaded=True
    )
