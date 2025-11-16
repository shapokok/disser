"""
Configuration file for Crop Disease Detection System
Modify these settings to customize the application
"""

import os

# =======================
# Server Configuration
# =======================

# Flask server settings
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5000        # Default port
DEBUG = True       # Enable debug mode (disable in production)

# =======================
# File Upload Settings
# =======================

# Maximum file size (in bytes)
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Upload directory
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads'))

# =======================
# Model Configuration
# =======================

# Models directory
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Results directory
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'heatmaps'))

# Default model for inference
DEFAULT_MODEL = 'efficientnet'

# Models to load at startup
MODELS_TO_LOAD = [
    ('baseline', 'baseline'),
    ('efficientnet', 'efficientnet'),
    ('mobilenet', 'mobilenet'),
    ('hybrid', 'hybrid')
]

# =======================
# Image Processing
# =======================

# Input image size for models (height, width)
IMAGE_SIZE = (224, 224)

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =======================
# Explainability Settings
# =======================

# Default explanation method
DEFAULT_EXPLANATION = 'gradcam'  # 'gradcam' or 'lime'

# Grad-CAM settings
GRADCAM_ALPHA = 0.4  # Overlay transparency (0-1)
GRADCAM_COLORMAP = 'jet'  # Matplotlib colormap

# LIME settings
LIME_NUM_SAMPLES = 1000  # Number of samples for LIME explanation

# =======================
# Performance Settings
# =======================

# Use GPU if available
USE_GPU = True

# Number of threads for CPU inference
NUM_THREADS = 4

# Batch size for batch processing
BATCH_SIZE = 8

# =======================
# API Settings
# =======================

# Enable CORS
ENABLE_CORS = True

# CORS allowed origins (use ['*'] for all)
CORS_ORIGINS = '*'

# API rate limiting (requests per minute)
RATE_LIMIT = 100

# =======================
# Logging Settings
# =======================

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = 'INFO'

# Log file path
LOG_FILE = os.path.join(os.path.dirname(__file__), 'app.log')

# =======================
# Frontend Settings
# =======================

# Frontend directory
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

# =======================
# Dataset Settings
# =======================

# Number of disease classes
NUM_CLASSES = 38

# Dataset types
DATASET_TYPES = ['controlled', 'field']

# =======================
# Development Settings
# =======================

# Enable detailed error messages
VERBOSE_ERRORS = DEBUG

# Save uploaded files permanently (for debugging)
SAVE_UPLOADS = True

# Save generated heatmaps
SAVE_HEATMAPS = True

# =======================
# Security Settings
# =======================

# Secret key for session management (change in production!)
SECRET_KEY = 'change-this-in-production-please'

# Allowed hosts (for production)
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# =======================
# Helper Functions
# =======================

def get_device():
    """Get PyTorch device based on configuration"""
    import torch
    if USE_GPU and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        UPLOAD_FOLDER,
        MODELS_DIR,
        RESULTS_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# =======================
# Validation
# =======================

def validate_config():
    """Validate configuration settings"""
    assert MAX_FILE_SIZE > 0, "MAX_FILE_SIZE must be positive"
    assert PORT > 0 and PORT < 65536, "PORT must be between 1 and 65535"
    assert IMAGE_SIZE[0] > 0 and IMAGE_SIZE[1] > 0, "IMAGE_SIZE must be positive"
    assert 0 <= GRADCAM_ALPHA <= 1, "GRADCAM_ALPHA must be between 0 and 1"
    assert LIME_NUM_SAMPLES > 0, "LIME_NUM_SAMPLES must be positive"
    assert NUM_CLASSES > 0, "NUM_CLASSES must be positive"

# Run validation
validate_config()
