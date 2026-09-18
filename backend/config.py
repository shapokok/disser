"""
Central configuration for the backend.

Every value can be overridden with an environment variable (prefix CROP_),
so the same code runs locally, in Docker and in CI without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Paths -----------------------------------------------------------------
MODELS_DIR = _path("CROP_MODELS_DIR", PROJECT_ROOT / "models")
DATA_DIR = _path("CROP_DATA_DIR", PROJECT_ROOT / "data")
UPLOAD_DIR = _path("CROP_UPLOAD_DIR", DATA_DIR / "uploads")
RESULTS_DIR = _path("CROP_RESULTS_DIR", PROJECT_ROOT / "results")
METRICS_DIR = RESULTS_DIR / "metrics"
FRONTEND_DIR = _path("CROP_FRONTEND_DIR", PROJECT_ROOT / "frontend")
PLANTVILLAGE_DIR = DATA_DIR / "PlantVillage"
DA_RESULTS_DIR = _path("CROP_DA_RESULTS_DIR", PROJECT_ROOT / "domain_adaptation_experiments" / "results")

# --- Server ------------------------------------------------------------------
HOST = os.environ.get("CROP_HOST", "0.0.0.0")
# 5000 is taken by AirPlay Receiver on macOS, so the default is 5001.
PORT = int(os.environ.get("CROP_PORT", os.environ.get("PORT", "5001")))
DEBUG = _bool("CROP_DEBUG", False)
CORS_ORIGINS = os.environ.get("CROP_CORS_ORIGINS", "*")

# --- Uploads -----------------------------------------------------------------
MAX_FILE_SIZE = int(os.environ.get("CROP_MAX_FILE_MB", "16")) * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "jfif", "webp", "bmp"}

# --- Models ------------------------------------------------------------------
# Which architectures to load at start-up (comma separated).
MODELS_TO_LOAD = [
    m.strip()
    for m in os.environ.get("CROP_MODELS", "baseline,efficientnet,mobilenet,hybrid").split(",")
    if m.strip()
]
# auto | cpu | cuda | mps
DEVICE = os.environ.get("CROP_DEVICE", "auto")
# A model whose validation accuracy is below this is shown as "needs training"
# and excluded from the ensemble.
TRAINED_ACCURACY_THRESHOLD = float(os.environ.get("CROP_TRAINED_THRESHOLD", "0.5"))

# Field-condition model produced by the domain adaptation experiments.
FIELD_MODEL_PATH = _path("CROP_FIELD_MODEL", MODELS_DIR / "field_mobilenet_da.pth")
FIELD_MODEL_INFO = MODELS_DIR / "field_model_info.json"

# --- Image processing --------------------------------------------------------
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Explainability ----------------------------------------------------------
GRADCAM_ALPHA = 0.45
# LIME perturbation samples: ~15 s on Apple GPU, ~1 min on CPU for EfficientNet.
LIME_NUM_SAMPLES = int(os.environ.get("CROP_LIME_SAMPLES", "300"))
# Size of the base64 images returned to the browser (longest side, px).
PREVIEW_MAX_SIDE = int(os.environ.get("CROP_PREVIEW_SIDE", "512"))

APP_VERSION = "2.0.0"
