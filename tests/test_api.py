"""
Backend API tests (pytest).

    pytest                       # fast tests + model tests when weights exist
    pytest -m "not models"       # only tests that need no weights (CI)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import config
from app import create_app

WEIGHTS = ROOT / "models" / "efficientnet_model.pth"
needs_models = pytest.mark.models
skip_without_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="models/efficientnet_model.pth not available")


@pytest.fixture(scope="session")
def app_no_models():
    return create_app(load_models=False)


@pytest.fixture(scope="session")
def client(app_no_models):
    return app_no_models.test_client()


@pytest.fixture(scope="session")
def model_client():
    app = create_app(load_models=True, models=["efficientnet", "baseline"])
    return app.test_client()


@pytest.fixture()
def leaf_image(tmp_path):
    """A small synthetic leaf-like JPEG."""
    img = Image.new("RGB", (256, 256), (40, 120, 40))
    for x in range(60, 200):
        for y in range(60, 200):
            if (x - 130) ** 2 + (y - 130) ** 2 < 60**2:
                img.putpixel((x, y), (150, 90, 30))
    p = tmp_path / "leaf.jpg"
    img.save(p, "JPEG")
    return p


# ------------------------------------------------------------------ meta endpoints (no weights)
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "online"
    assert data["total_classes"] == 38


def test_index_serves_frontend(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<html" in r.data
    assert client.get("/analyze").status_code == 200
    assert client.get("/stats").status_code == 200
    assert client.get("/css/style.css").status_code == 200


def test_static_traversal_blocked(client):
    r = client.get("/../backend/app.py")
    assert r.status_code == 404


def test_classes(client):
    data = client.get("/api/classes").get_json()
    assert data["total"] == 38
    first = data["classes"][0]
    assert {"class", "class_ru", "plant", "disease", "healthy"} <= set(first)
    assert "Tomato" in data["grouped_by_plant"]


def test_models_listing(client):
    data = client.get("/api/models").get_json()
    assert data["success"]
    assert isinstance(data["models"], list)
    assert "field_model" in data


def test_treatment_every_class_both_languages(client):
    classes = client.get("/api/classes").get_json()["classes"]
    for c in classes:
        for lang in ("ru", "en"):
            data = client.get(f"/api/treatment/{c['name']}?lang={lang}").get_json()
            tr = data["treatment"]
            assert tr["lang"] == lang
            assert tr["generic"] is False, f"{c['name']} has no {lang} treatment entry"
            assert tr["treatments"] and tr["prevention"]


def test_treatment_unknown_class_is_generic(client):
    tr = client.get("/api/treatment/Unknown___thing?lang=en").get_json()["treatment"]
    assert tr["generic"] is True
    assert tr["treatments"]


def test_predict_requires_image(client):
    r = client.post("/api/predict", json={})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_predict_missing_image(client):
    r = client.post("/api/predict", json={"image_path": "nope.jpg"})
    assert r.status_code == 404


def test_upload_rejects_bad_extension(client):
    data = {"files": (io.BytesIO(b"hello"), "notes.txt")}
    r = client.post("/api/upload", data=data, content_type="multipart/form-data")
    body = r.get_json()
    assert body["count"] == 0
    assert body["errors"]


def test_upload_rejects_non_image(client):
    data = {"files": (io.BytesIO(b"not really a jpeg"), "fake.jpg")}
    body = client.post("/api/upload", data=data, content_type="multipart/form-data").get_json()
    assert body["count"] == 0
    assert "not a valid image" in body["errors"][0]


def test_upload_accepts_image(client, leaf_image):
    with open(leaf_image, "rb") as f:
        body = client.post(
            "/api/upload", data={"files": (f, "leaf.jfif")}, content_type="multipart/form-data"
        ).get_json()
    assert body["count"] == 1
    saved = config.UPLOAD_DIR / body["uploaded"][0]["saved_name"]
    assert saved.exists()
    assert client.get(body["uploaded"][0]["url"]).status_code == 200
    saved.unlink()


def _fake_result():
    return {
        "image_name": "leaf.jpg",
        "model": "efficientnet",
        "model_label": "EfficientNet-B0",
        "explanation": "gradcam",
        "dataset_type": "controlled",
        "inference_time_ms": 12.3,
        "timestamp": "2026-09-18T12:00:00",
        "prediction": {
            "class_raw": "Tomato___Late_blight",
            "class": "Tomato — Late blight",
            "class_ru": "Томат — Фитофтороз",
            "confidence": 0.91,
            "confidence_percent": "91.00%",
        },
        "top_predictions": [
            {
                "class_raw": "Tomato___Late_blight",
                "class": "Tomato — Late blight",
                "class_ru": "Томат — Фитофтороз",
                "confidence": 0.91,
                "confidence_percent": "91.00%",
            },
            {
                "class_raw": "Tomato___Early_blight",
                "class": "Tomato — Early blight",
                "class_ru": "Томат — Альтернариоз",
                "confidence": 0.05,
                "confidence_percent": "5.00%",
            },
        ],
        "treatment": {
            "disease_name": "Фитофтороз",
            "severity": "критическая",
            "symptoms": "…",
            "treatments": ["a"],
            "prevention": ["b"],
            "organic_options": ["c"],
        },
        "images": {},
    }


@pytest.mark.parametrize(
    "fmt,mime",
    [("csv", "text/csv"), ("json", "application/json"), ("excel", "spreadsheetml"), ("pdf", "application/pdf")],
)
def test_export_formats(client, fmt, mime):
    r = client.post(f"/api/export/{fmt}?lang=ru", json={"results": [_fake_result()]})
    assert r.status_code == 200, r.data[:200]
    assert mime in r.headers["Content-Type"]
    assert len(r.data) > 100
    if fmt == "csv":
        assert "Фитофтороз" in r.data.decode("utf-8-sig")


def test_export_requires_results(client):
    assert client.post("/api/export/csv", json={}).status_code == 400
    assert client.post("/api/export/xml", json={"results": [_fake_result()]}).status_code == 404


def test_comparison_export(client):
    payload = {
        "image_name": "leaf.jpg",
        "comparisons": {
            "efficientnet": {
                "label": "EfficientNet-B0",
                "class": "Tomato — Late blight",
                "class_ru": "Томат — Фитофтороз",
                "confidence": 0.9,
                "confidence_percent": "90.00%",
                "inference_time_ms": 10,
                "trained": True,
            }
        },
        "agreement": {
            "class": {"class": "Tomato — Late blight", "class_ru": "Томат — Фитофтороз"},
            "models_agreeing": 1,
            "models_total": 1,
        },
    }
    for fmt in ("csv", "excel", "pdf"):
        r = client.post(f"/api/export/comparison/{fmt}?lang=en", json=payload)
        assert r.status_code == 200, fmt


def test_validation_endpoint_shape(client):
    r = client.get("/api/validation/all?lang=en")
    assert r.status_code == 200
    reports = r.get_json()["reports"]
    for rep in reports.values():
        assert len(rep["confusion_matrix"]) == 38
        assert len(rep["per_class"]) == 38
        assert rep["overall"]["accuracy"] <= 1.0


def test_stats_and_research(client):
    assert client.get("/api/stats").get_json()["success"]
    assert client.get("/api/research").get_json()["success"]
    assert client.get("/api/dataset_info").get_json()["success"]


# ------------------------------------------------------------------ inference (needs weights)
@needs_models
@skip_without_weights
def test_predict_gradcam_and_compare(model_client, leaf_image):
    with open(leaf_image, "rb") as f:
        up = model_client.post(
            "/api/upload", data={"files": (f, "leaf.jpg")}, content_type="multipart/form-data"
        ).get_json()
    name = up["uploaded"][0]["saved_name"]
    try:
        r = model_client.post(
            "/api/predict", json={"image_path": name, "model": "efficientnet", "explanation": "gradcam", "lang": "ru"}
        )
        assert r.status_code == 200, r.data[:300]
        data = r.get_json()
        assert data["success"]
        assert 0 <= data["prediction"]["confidence"] <= 1
        assert set(data["images"]) == {"original", "heatmap", "overlay"}
        assert len(data["top_predictions"]) == 5
        assert data["treatment"]["lang"] == "ru"

        r = model_client.post("/api/predict", json={"image_path": name, "model": "baseline", "explanation": "none"})
        assert r.status_code == 200
        assert set(r.get_json()["images"]) == {"original"}

        r = model_client.post("/api/compare", json={"image_path": name})
        assert r.status_code == 200
        cmp_ = r.get_json()
        assert set(cmp_["comparisons"]) == {"efficientnet", "baseline"}
        assert cmp_["agreement"]["models_total"] == 2

        r = model_client.post("/api/predict", json={"image_path": name, "model": "nope"})
        assert r.status_code == 400
    finally:
        (config.UPLOAD_DIR / name).unlink(missing_ok=True)


@needs_models
@skip_without_weights
def test_ensemble_when_enough_trained_models(model_client, leaf_image):
    with open(leaf_image, "rb") as f:
        up = model_client.post(
            "/api/upload", data={"files": (f, "leaf.jpg")}, content_type="multipart/form-data"
        ).get_json()
    name = up["uploaded"][0]["saved_name"]
    try:
        r = model_client.post("/api/ensemble", json={"image_path": name, "explanation": "none"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["ensemble"]["models_used"]
        assert data["ensemble"]["uncertainty_metrics"]["uncertainty_level"] in {"low", "medium", "high"}
    finally:
        (config.UPLOAD_DIR / name).unlink(missing_ok=True)
