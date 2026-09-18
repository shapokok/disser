"""Analysis journal: SQLite store and the /api/history endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import config  # noqa: E402
from history import History  # noqa: E402


def fake_result(i: int, healthy: bool = False, model: str = "efficientnet") -> dict:
    cls = "Tomato___healthy" if healthy else "Tomato___Late_blight"
    return {
        "timestamp": f"2026-09-{10 + i % 5:02d}T12:00:00",
        "image_name": f"leaf_{i}.jpg",
        "model": model,
        "model_label": "EfficientNet-B0",
        "explanation": "gradcam",
        "dataset_type": "controlled",
        "inference_time_ms": 42.0,
        "prediction": {"class_raw": cls, "plant": "Tomato", "confidence": 0.9, "healthy": healthy},
        "top_predictions": [],
        "images": {"original": "AAAA"},
    }


def test_history_store(tmp_path):
    h = History(tmp_path / "h.sqlite")
    ids = [
        h.add(fake_result(i, healthy=i % 3 == 0, model="baseline" if i % 2 else "efficientnet"), "thumb")
        for i in range(7)
    ]
    assert ids == list(range(1, 8))
    items, total = h.list(limit=3)
    assert total == 7 and len(items) == 3 and items[0]["id"] == 7
    assert "result_json" not in items[0] and items[0]["thumbnail"] == "thumb"
    assert h.list(model="baseline")[1] == 3
    assert h.list(healthy=True)[1] == 3
    assert h.list(query="Late_blight")[1] == 4
    st = h.stats()
    assert st["total"] == 7 and st["healthy"] == 3
    assert st["by_class"][0]["class_raw"] == "Tomato___Late_blight" and st["by_class"][0]["n"] == 4
    assert sum(d["n"] for d in st["by_day"]) == 7
    full = h.get(1)
    assert full["result"]["image_name"] == "leaf_0.jpg" and "images" not in full["result"]
    assert h.delete(1) and not h.delete(1)
    assert h.clear() == 6 and h.stats()["total"] == 0


@pytest.fixture()
def hist_client(tmp_path, monkeypatch):
    from app import create_app

    monkeypatch.setattr(config, "HISTORY_DB", tmp_path / "api.sqlite")
    app = create_app(load_models=False)
    app.extensions["history"].add(fake_result(1), None)
    app.extensions["history"].add(fake_result(2, healthy=True), None)
    return app.test_client()


def test_history_api(hist_client):
    data = hist_client.get("/api/history").get_json()
    assert data["total"] == 2
    assert data["items"][0]["prediction"]["class_ru"] == "Томат — Здоровое растение"
    assert hist_client.get("/api/history?healthy=0").get_json()["total"] == 1
    st = hist_client.get("/api/history/stats").get_json()
    assert st["total"] == 2 and st["healthy_share"] == 0.5
    assert st["by_class"][0]["class"]
    item = hist_client.get("/api/history/1").get_json()["item"]
    assert item["result"]["model"] == "efficientnet"
    assert hist_client.get("/api/history/999").status_code == 404
    assert hist_client.delete("/api/history/1").get_json()["success"] is True
    assert hist_client.delete("/api/history").get_json()["deleted"] == 1
    assert hist_client.get("/api/history").get_json()["total"] == 0
    assert hist_client.get("/history").status_code == 200
