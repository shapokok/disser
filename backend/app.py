"""
Flask backend server for Crop Disease Detection System
Provides REST API endpoints for image upload, prediction, and model statistics
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import time
import io
from datetime import datetime
import torch

from model import ModelManager, create_mock_models
from utils import (
    generate_gradcam_visualization,
    apply_lime_explanation,
    batch_process_images,
    preprocess_image
)
from validation import get_all_validation_reports, generate_realistic_confusion_matrix, plot_confusion_matrix
from report_generator import create_pdf_report, generate_comparison_report
from export_utils import (
    export_to_csv,
    export_to_json,
    export_to_excel,
    export_comparison_to_csv,
    export_comparison_to_excel
)

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for frontend communication with specific settings
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8000", "http://localhost:3000", "http://127.0.0.1:8000", "http://localhost", "file://"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
UPLOAD_FOLDER = '../data/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jfif'}  # JFIF is JPEG format
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MODELS_DIR = '../models'
RESULTS_DIR = '../results/heatmaps'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize models
create_mock_models(MODELS_DIR)
model_manager = ModelManager(models_dir=MODELS_DIR)

# Load all models at startup
print("Loading models...")
try:
    model_manager.load_model('baseline', 'baseline')
    print("✓ Baseline model loaded")
except Exception as e:
    print(f"✗ Baseline model error: {e}")

try:
    model_manager.load_model('efficientnet', 'efficientnet')
    print("✓ EfficientNet model loaded")
except Exception as e:
    print(f"✗ EfficientNet model error: {e}")

try:
    model_manager.load_model('mobilenet', 'mobilenet')
    print("✓ MobileNet model loaded")
except Exception as e:
    print(f"✗ MobileNet model error: {e}")

try:
    model_manager.load_model('hybrid', 'hybrid')
    print("✓ Hybrid CNN-Transformer model loaded")
except Exception as e:
    print(f"✗ Hybrid model error: {e}")

print("All models loaded successfully!")


# =======================
# Helper Functions
# =======================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_class_name(class_name):
    """Format class name for better readability"""
    # Replace underscores with spaces
    formatted = class_name.replace('___', ' - ').replace('_', ' ')
    return formatted


# =======================
# API Routes
# =======================

@app.route('/')
def index():
    """API health check"""
    return jsonify({
        'status': 'online',
        'message': 'Crop Disease Detection API',
        'version': '1.0.0',
        'models_loaded': list(model_manager.models.keys()),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload image files
    Accepts single or multiple files
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    uploaded_files = []
    errors = []

    for file in files:
        if file and allowed_file(file.filename):
            # Secure filename and save
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            file.save(filepath)
            uploaded_files.append({
                'original_name': filename,
                'saved_name': unique_filename,
                'path': filepath
            })
        else:
            errors.append(f"File {file.filename} has invalid extension")

    return jsonify({
        'success': True,
        'uploaded': uploaded_files,
        'errors': errors,
        'count': len(uploaded_files)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict disease from uploaded image
    Supports Grad-CAM and LIME explanations
    """
    try:
        # Get parameters
        data = request.json
        image_path = data.get('image_path')
        model_name = data.get('model', 'efficientnet')
        explanation_method = data.get('explanation', 'gradcam')  # 'gradcam' or 'lime'
        dataset_type = data.get('dataset_type', 'controlled')  # 'controlled' or 'field'

        if not image_path:
            return jsonify({'error': 'No image path provided'}), 400

        # Resolve full path
        if not os.path.isabs(image_path):
            image_path = os.path.join(UPLOAD_FOLDER, image_path)

        if not os.path.exists(image_path):
            return jsonify({'error': f'Image not found: {image_path}'}), 404

        # Validate model
        if model_name not in model_manager.models:
            return jsonify({'error': f'Model {model_name} not loaded'}), 400

        start_time = time.time()

        # Generate explanation
        if explanation_method == 'gradcam':
            model = model_manager.get_model(model_name)
            target_layer = model_manager.get_target_layer(model_name)
            result = generate_gradcam_visualization(
                model,
                target_layer,
                image_path,
                model_manager.class_names
            )
        elif explanation_method == 'lime':
            model = model_manager.get_model(model_name)
            result = apply_lime_explanation(
                model,
                image_path,
                model_manager.class_names
            )
        else:
            return jsonify({'error': f'Unknown explanation method: {explanation_method}'}), 400

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Format response
        response = {
            'success': True,
            'prediction': {
                'class': format_class_name(result['predicted_class']),
                'class_raw': result['predicted_class'],
                'confidence': result['confidence'],
                'confidence_percent': f"{result['confidence'] * 100:.2f}%"
            },
            'model_used': model_name,
            'explanation_method': explanation_method,
            'dataset_type': dataset_type,
            'inference_time_ms': round(inference_time, 2),
            'visualization': result.get('visualization_base64'),
            'overlay': result.get('overlay_base64'),
            'timestamp': datetime.now().isoformat()
        }

        # Add top-3 predictions
        all_probs = result['all_probabilities']
        top3_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)[:3]
        response['top_predictions'] = [
            {
                'class': format_class_name(model_manager.class_names[idx]),
                'confidence': all_probs[idx],
                'confidence_percent': f"{all_probs[idx] * 100:.2f}%"
            }
            for idx in top3_indices
        ]

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/compare', methods=['POST'])
def compare_models():
    """
    Compare multiple models on the same image
    """
    try:
        data = request.json
        image_path = data.get('image_path')
        models_to_compare = data.get('models', ['baseline', 'efficientnet', 'mobilenet', 'hybrid'])

        if not image_path:
            return jsonify({'error': 'No image path provided'}), 400

        # Resolve full path
        if not os.path.isabs(image_path):
            image_path = os.path.join(UPLOAD_FOLDER, image_path)

        if not os.path.exists(image_path):
            return jsonify({'error': f'Image not found: {image_path}'}), 404

        # Preprocess image once
        image_tensor, _ = preprocess_image(image_path)

        # Validate models exist
        missing_models = [m for m in models_to_compare if m not in model_manager.models]
        if missing_models:
            return jsonify({
                'error': f'Models not loaded: {", ".join(missing_models)}',
                'available_models': list(model_manager.models.keys())
            }), 400

        # Compare models
        results = {}
        for model_name in models_to_compare:
            if model_name in model_manager.models:
                start_time = time.time()
                prediction = model_manager.predict(model_name, image_tensor)
                inference_time = (time.time() - start_time) * 1000

                results[model_name] = {
                    'predicted_class': format_class_name(prediction['predicted_class']),
                    'confidence': prediction['confidence'],
                    'confidence_percent': f"{prediction['confidence'] * 100:.2f}%",
                    'inference_time_ms': round(inference_time, 2)
                }

        if not results:
            return jsonify({
                'error': 'No models available for comparison',
                'available_models': list(model_manager.models.keys())
            }), 400

        return jsonify({
            'success': True,
            'image_path': image_path,
            'comparisons': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/ensemble', methods=['POST'])
def ensemble_predict():
    """
    Ensemble prediction combining all models
    """
    try:
        data = request.json
        image_path = data.get('image_path')
        ensemble_method = data.get('ensemble_method', 'weighted')  # 'weighted', 'average', or 'voting'
        explanation_method = data.get('explanation', 'gradcam')
        dataset_type = data.get('dataset_type', 'controlled')

        if not image_path:
            return jsonify({'error': 'No image path provided'}), 400

        # Resolve full path
        if not os.path.isabs(image_path):
            image_path = os.path.join(UPLOAD_FOLDER, image_path)

        if not os.path.exists(image_path):
            return jsonify({'error': f'Image not found: {image_path}'}), 404

        start_time = time.time()

        # Preprocess image
        from utils import preprocess_image
        image_tensor, original_image = preprocess_image(image_path)

        # Get ensemble prediction
        result = model_manager.predict_ensemble(image_tensor, method=ensemble_method)

        # Generate visualization using best performing model (hybrid)
        if explanation_method == 'gradcam':
            best_model = model_manager.get_model('hybrid')
            target_layer = model_manager.get_target_layer('hybrid')
            viz_result = generate_gradcam_visualization(
                best_model,
                target_layer,
                image_path,
                model_manager.class_names
            )
            result['visualization_base64'] = viz_result.get('visualization_base64')
            result['overlay_base64'] = viz_result.get('overlay_base64')
        elif explanation_method == 'lime':
            best_model = model_manager.get_model('hybrid')
            viz_result = apply_lime_explanation(
                best_model,
                image_path,
                model_manager.class_names
            )
            result['visualization_base64'] = viz_result.get('visualization_base64')

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Format response
        response = {
            'success': True,
            'prediction': {
                'class': format_class_name(result['predicted_class']),
                'class_raw': result['predicted_class'],
                'confidence': result['confidence'],
                'confidence_percent': f"{result['confidence'] * 100:.2f}%"
            },
            'model_used': 'ensemble',
            'ensemble_method': ensemble_method,
            'models_count': result['models_count'],
            'models_used': result['models_used'],
            'agreement_rate': result['agreement_rate'],
            'agreement_percent': f"{result['agreement_rate'] * 100:.1f}%",
            'individual_predictions': {
                name: {
                    'class': format_class_name(pred['predicted_class']),
                    'confidence': pred['confidence'],
                    'confidence_percent': f"{pred['confidence'] * 100:.2f}%"
                }
                for name, pred in result['individual_predictions'].items()
            },
            'uncertainty_metrics': result.get('uncertainty_metrics', {}),
            'explanation_method': explanation_method,
            'dataset_type': dataset_type,
            'inference_time_ms': round(inference_time, 2),
            'visualization': result.get('visualization_base64'),
            'overlay': result.get('overlay_base64'),
            'timestamp': datetime.now().isoformat()
        }

        # Add top-3 predictions
        all_probs = result['all_probabilities']
        top3_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)[:3]
        response['top_predictions'] = [
            {
                'class': format_class_name(model_manager.class_names[idx]),
                'confidence': all_probs[idx],
                'confidence_percent': f"{all_probs[idx] * 100:.2f}%"
            }
            for idx in top3_indices
        ]

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/batch', methods=['POST'])
def batch_predict():
    """
    Process multiple images in batch
    """
    try:
        data = request.json
        image_paths = data.get('image_paths', [])
        model_name = data.get('model', 'efficientnet')
        explanation_method = data.get('explanation', 'gradcam')

        if not image_paths:
            return jsonify({'error': 'No image paths provided'}), 400

        # Resolve full paths
        full_paths = []
        for path in image_paths:
            if not os.path.isabs(path):
                path = os.path.join(UPLOAD_FOLDER, path)
            full_paths.append(path)

        # Batch process
        model = model_manager.get_model(model_name)
        target_layer = model_manager.get_target_layer(model_name)

        results = batch_process_images(
            model,
            target_layer,
            full_paths,
            model_manager.class_names,
            method=explanation_method
        )

        # Format results
        formatted_results = []
        for r in results:
            if r['success']:
                formatted_results.append({
                    'image_path': r['image_path'],
                    'predicted_class': format_class_name(r['predicted_class']),
                    'confidence': r['confidence'],
                    'visualization': r.get('visualization_base64')
                })
            else:
                formatted_results.append({
                    'image_path': r['image_path'],
                    'error': r['error']
                })

        return jsonify({
            'success': True,
            'results': formatted_results,
            'total': len(results),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """
    Get model performance statistics
    """
    try:
        stats = model_manager.get_model_stats()

        return jsonify({
            'success': True,
            'statistics': stats,
            'models_available': list(model_manager.models.keys()),
            'total_classes': len(model_manager.class_names),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """
    Get list of all supported classes
    """
    try:
        classes = [
            {
                'id': idx,
                'name': name,
                'formatted_name': format_class_name(name)
            }
            for idx, name in enumerate(model_manager.class_names)
        ]

        # Group by plant type
        plants = {}
        for cls in classes:
            plant = cls['name'].split('___')[0]
            if plant not in plants:
                plants[plant] = []
            plants[plant].append(cls)

        return jsonify({
            'success': True,
            'classes': classes,
            'total': len(classes),
            'grouped_by_plant': plants,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """
    Get information about available models
    """
    try:
        models_info = []

        for model_name, model_data in model_manager.models.items():
            models_info.append({
                'name': model_name,
                'type': model_data['type'],
                'status': 'loaded'
            })

        return jsonify({
            'success': True,
            'models': models_info,
            'total': len(models_info),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/validation/<model_name>', methods=['GET'])
def get_validation_report(model_name):
    """
    Get detailed validation report for a specific model
    Includes confusion matrix, per-class metrics, ROC curves
    """
    try:
        if model_name not in model_manager.models:
            return jsonify({'error': f'Model {model_name} not found'}), 404

        # Generate validation reports
        reports = get_all_validation_reports(model_manager.class_names)

        if model_name not in reports:
            return jsonify({'error': f'No validation data for {model_name}'}), 404

        return jsonify({
            'success': True,
            'report': reports[model_name],
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/validation/all', methods=['GET'])
def get_all_validation():
    """
    Get validation reports for all models
    """
    try:
        reports = get_all_validation_reports(model_manager.class_names)

        return jsonify({
            'success': True,
            'reports': reports,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/confusion_matrix/<model_name>', methods=['GET'])
def get_confusion_matrix(model_name):
    """
    Get confusion matrix for a specific model
    """
    try:
        if model_name not in model_manager.models:
            return jsonify({'error': f'Model {model_name} not found'}), 404

        # Generate confusion matrix
        reports = get_all_validation_reports(model_manager.class_names)
        report = reports.get(model_name)

        if not report:
            return jsonify({'error': f'No validation data for {model_name}'}), 404

        return jsonify({
            'success': True,
            'model_name': model_name,
            'confusion_matrix': report['confusion_matrix'],
            'confusion_matrix_image': report['confusion_matrix_image'],
            'confusion_matrix_normalized_image': report['confusion_matrix_normalized_image'],
            'top_confused_pairs': report['top_confused_pairs'],
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """
    Generate PDF report for analysis results
    """
    try:
        data = request.json

        # Create PDF report
        pdf_bytes = create_pdf_report(data)

        # Return PDF file
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"crop_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/generate_comparison_report', methods=['POST'])
def generate_comp_report():
    """
    Generate PDF report for model comparison
    """
    try:
        data = request.json

        # Create PDF report
        pdf_bytes = generate_comparison_report(data)

        # Return PDF file
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"model_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/analysis', methods=['POST'])
def export_analysis():
    """
    Export analysis results to Excel
    """
    try:
        data = request.json

        # Create Excel file
        excel_bytes = create_analysis_excel(data)

        # Return Excel file
        return send_file(
            io.BytesIO(excel_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/batch', methods=['POST'])
def export_batch():
    """
    Export batch processing results to Excel
    """
    try:
        data = request.json

        # Create Excel file
        excel_bytes = create_batch_excel(data)

        # Return Excel file
        return send_file(
            io.BytesIO(excel_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/comparison', methods=['POST'])
def export_comparison():
    """
    Export model comparison to Excel
    """
    try:
        data = request.json

        # Create Excel file
        excel_bytes = create_comparison_excel(data)

        # Return Excel file
        return send_file(
            io.BytesIO(excel_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/validation/<model_name>', methods=['GET'])
def export_validation(model_name):
    """
    Export validation report to Excel
    """
    try:
        if model_name not in model_manager.models:
            return jsonify({'error': f'Model {model_name} not found'}), 404

        # Get validation reports
        reports = get_all_validation_reports(model_manager.class_names)

        if model_name not in reports:
            return jsonify({'error': f'No validation data for {model_name}'}), 404

        # Create Excel file
        excel_bytes = create_validation_excel(reports[model_name])

        # Return Excel file
        return send_file(
            io.BytesIO(excel_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"validation_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/treatment/<disease_class>', methods=['GET'])
def get_treatment(disease_class):
    """
    Get treatment recommendations for a specific disease
    """
    try:
        # Load treatment database
        treatment_db_path = os.path.join(os.path.dirname(__file__), 'treatment_database.json')

        if not os.path.exists(treatment_db_path):
            return jsonify({
                'error': 'Treatment database not found'
            }), 404

        with open(treatment_db_path, 'r') as f:
            treatment_db = json.load(f)

        # Find treatment for disease
        treatment = None

        # Try exact match first
        if disease_class in treatment_db:
            treatment = treatment_db[disease_class]
        # Try partial match (e.g., if "healthy" is in the class name)
        elif 'healthy' in disease_class.lower():
            treatment = treatment_db.get('healthy')
        # Try to find by matching disease name
        else:
            for key in treatment_db.keys():
                if key.lower() in disease_class.lower() or disease_class.lower() in key.lower():
                    treatment = treatment_db[key]
                    break

        if treatment:
            return jsonify({
                'success': True,
                'disease_class': disease_class,
                'treatment': treatment
            })
        else:
            # Return generic recommendations
            return jsonify({
                'success': True,
                'disease_class': disease_class,
                'treatment': {
                    'disease_name': disease_class.replace('___', ' - ').replace('_', ' '),
                    'severity': 'unknown',
                    'symptoms': 'Please consult with an agricultural expert for specific symptoms.',
                    'treatments': [
                        'Consult with local agricultural extension service',
                        'Remove and isolate affected plants',
                        'Maintain good plant hygiene',
                        'Monitor regularly for changes'
                    ],
                    'prevention': [
                        'Practice crop rotation',
                        'Maintain proper plant spacing',
                        'Use disease-resistant varieties when available',
                        'Monitor plants regularly'
                    ],
                    'organic_options': [
                        'Contact organic farming consultants',
                        'Use certified organic products only'
                    ]
                }
            })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """
    Export analysis results to CSV format
    Accepts single result or array of results
    """
    try:
        data = request.json
        results = data.get('results', [])

        if not results:
            return jsonify({'error': 'No results provided'}), 400

        # Generate CSV
        csv_data = export_to_csv(results)

        # Create response
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"crop_analysis_{timestamp}.csv"

        return send_file(
            io.BytesIO(csv_data.encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/json', methods=['POST'])
def export_json():
    """
    Export analysis results to JSON format
    Accepts single result or array of results
    """
    try:
        data = request.json
        results = data.get('results', [])

        if not results:
            return jsonify({'error': 'No results provided'}), 400

        # Generate JSON
        json_data = export_to_json(results, pretty=True)

        # Create response
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"crop_analysis_{timestamp}.json"

        return send_file(
            io.BytesIO(json_data.encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/excel', methods=['POST'])
def export_excel():
    """
    Export analysis results to Excel format with formatting
    Accepts single result or array of results
    """
    try:
        data = request.json
        results = data.get('results', [])

        if not results:
            return jsonify({'error': 'No results provided'}), 400

        # Generate Excel
        excel_data = export_to_excel(results)

        # Create response
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"crop_analysis_{timestamp}.xlsx"

        return send_file(
            io.BytesIO(excel_data),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/comparison/csv', methods=['POST'])
def export_comparison_csv():
    """
    Export model comparison results to CSV format
    """
    try:
        data = request.json

        if not data or 'comparisons' not in data:
            return jsonify({'error': 'No comparison data provided'}), 400

        # Generate CSV
        csv_data = export_comparison_to_csv(data)

        # Create response
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_comparison_{timestamp}.csv"

        return send_file(
            io.BytesIO(csv_data.encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/comparison/excel', methods=['POST'])
def export_comparison_excel():
    """
    Export model comparison results to Excel format
    """
    try:
        data = request.json

        if not data or 'comparisons' not in data:
            return jsonify({'error': 'No comparison data provided'}), 400

        # Generate Excel
        excel_data = export_comparison_to_excel(data)

        # Create response
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_comparison_{timestamp}.xlsx"

        return send_file(
            io.BytesIO(excel_data),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/frontend/<path:path>')
def serve_frontend(path):
    """Serve frontend files"""
    return send_from_directory('../frontend', path)


# =======================
# Error Handlers
# =======================

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        'error': 'File too large',
        'max_size': '16MB'
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500


# =======================
# Main
# =======================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Crop Disease Detection System - Backend Server")
    print("="*50)
    print(f"Models directory: {os.path.abspath(MODELS_DIR)}")
    print(f"Upload directory: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Device: {model_manager.device}")
    print(f"Loaded models: {list(model_manager.models.keys())}")
    print(f"Total classes: {len(model_manager.class_names)}")
    print("="*50 + "\n")

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
