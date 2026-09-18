"""Unit tests for backend helpers and the domain adaptation utilities (no model weights needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "domain_adaptation_experiments"))

from labels import describe_class, format_class
from model import (
    BaselineCNN,
    EfficientNetModel,
    HybridCNNTransformer,
    MaskedModel,
    MobileNetModel,
    count_parameters,
)
from treatments import database, get_treatment


def test_labels_cover_all_classes():
    import json

    classes = json.loads((ROOT / "models" / "class_names.json").read_text())
    labels = json.loads((ROOT / "backend" / "class_labels.json").read_text(encoding="utf-8"))
    assert set(classes) == set(labels)
    assert format_class("Tomato___Late_blight", "ru") == "Томат — Фитофтороз"
    assert format_class("Tomato___Late_blight", "en") == "Tomato — Late blight"
    assert describe_class("Apple___healthy")["healthy"] is True


def test_treatment_database_complete():
    import json

    classes = json.loads((ROOT / "models" / "class_names.json").read_text())
    db = database()
    assert set(classes) == set(db)
    for name, entry in db.items():
        for lang in ("ru", "en"):
            assert entry[lang]["treatments"], f"{name}/{lang}"
    assert get_treatment("Potato___Late_blight", "ru")["disease_name"] == "Фитофтороз"


@pytest.mark.parametrize("factory", [BaselineCNN, EfficientNetModel, MobileNetModel])
def test_architectures_forward(factory):
    model = factory(num_classes=38, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 38)
    assert count_parameters(model) > 0


def test_hybrid_forward_shape():
    model = HybridCNNTransformer(num_classes=38, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 38)


def test_masked_model_restricts_classes():
    model = MaskedModel(BaselineCNN(38).eval(), [7, 9, 25, 30], 38).eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.zeros(1, 3, 224, 224)), dim=1)[0]
    assert probs[[7, 9, 25, 30]].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert probs[0].item() == 0.0


def test_gradcam_on_small_cnn():
    from explain import GradCAM

    model = BaselineCNN(38).eval()
    cam, probs, idx = GradCAM(model, model.features)(torch.rand(1, 3, 224, 224))
    assert cam.shape == (14, 14)
    assert 0 <= cam.min() <= cam.max() <= 1
    assert probs.shape == (38,)
    assert 0 <= idx < 38


def test_da_metrics_and_ci():
    from da import common as C

    lo, hi = C.wilson_ci(50, 100)
    assert lo < 0.5 < hi
    labels = np.array([7, 7, 9, 25, 30, 30])
    preds = np.array([7, 9, 9, 25, 30, 3])
    m = C.compute_metrics(labels, preds)
    assert m["n"] == 6 and m["correct"] == 4
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert len(m["confusion_matrix"]) == 4 and len(m["confusion_matrix"][0]) == 5  # + "other"
    probs = np.zeros((2, 38))
    probs[0, 3] = 0.9
    probs[0, 7] = 0.05
    probs[1, 30] = 0.6
    assert C.restricted_argmax(probs, C.FOCUS_IDX).tolist() == [7, 30]


def test_da_stratified_split_is_deterministic(tmp_path, monkeypatch):
    from da import common as C

    root = tmp_path / "plantdoc"
    for split, n in (("train", 20), ("test", 4)):
        for folder in C.FOCUS_CLASSES:
            d = root / split / folder
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{i}.jpg").write_bytes(b"x")
    monkeypatch.setattr(C, "PLANTDOC_DIR", root)
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path / "results")
    a = C.make_splits()
    b = C.make_splits()
    assert a == b
    assert len(a["adapt"]) == 64 and len(a["dev"]) == 16 and len(a["test"]) == 16
    assert not set(map(tuple, a["adapt"])) & set(map(tuple, a["dev"]))


def test_field_tta_probs_restricted_to_covered_classes(tmp_path):
    from PIL import Image

    import config
    from field_model import FieldAdapter
    from model import MobileNetModel

    ckpt = tmp_path / "field.pth"
    torch.save(MobileNetModel(38, pretrained=False).state_dict(), ckpt)
    adapter = FieldAdapter([f"c{i}" for i in range(38)], torch.device("cpu"), path=ckpt, info_path=tmp_path / "none.json")
    probs = adapter.tta_probs(Image.new("RGB", (300, 260), (40, 120, 40)))
    assert probs.shape == (38,)
    assert probs[[7, 9, 25, 30]].sum() == pytest.approx(1.0, abs=1e-5)
    assert config.FIELD_TTA in (True, False)
