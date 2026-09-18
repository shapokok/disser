"""
Model architectures and the ModelManager used by the API and the scripts.

The architectures must stay byte-compatible with the saved state dicts in
models/*.pth, so their module names are unchanged.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class BaselineCNN(nn.Module):
    """Small four-block CNN trained from scratch (reference model)."""

    def __init__(self, num_classes=38, pretrained=False):  # pretrained is ignored: trained from scratch
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _weights_or_none(enum_cls, pretrained):
    if not pretrained:
        return None
    try:
        return enum_cls.IMAGENET1K_V1
    except Exception:  # pragma: no cover - offline fallback
        return None


class EfficientNetModel(nn.Module):
    """EfficientNet-B0 (ImageNet) with a new classifier head."""

    def __init__(self, num_classes=38, pretrained=True):
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights

        try:
            self.backbone = models.efficientnet_b0(weights=_weights_or_none(EfficientNet_B0_Weights, pretrained))
        except Exception as e:  # download failure -> random init
            print(f"Warning: could not load EfficientNet ImageNet weights ({e}); using random init")
            self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))

    def forward(self, x):
        return self.backbone(x)


class MobileNetModel(nn.Module):
    """MobileNet-V2 (ImageNet) with a new classifier head."""

    def __init__(self, num_classes=38, pretrained=True):
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights

        try:
            self.backbone = models.mobilenet_v2(weights=_weights_or_none(MobileNet_V2_Weights, pretrained))
        except Exception as e:
            print(f"Warning: could not load MobileNet ImageNet weights ({e}); using random init")
            self.backbone = models.mobilenet_v2(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))

    def forward(self, x):
        return self.backbone(x)


class HybridCNNTransformer(nn.Module):
    """ResNet-50 feature extractor followed by a 2-layer Transformer encoder over the 7x7 feature grid."""

    def __init__(self, num_classes=38, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet50_Weights

        try:
            resnet = models.resnet50(weights=_weights_or_none(ResNet50_Weights, pretrained))
        except Exception as e:
            print(f"Warning: could not load ResNet-50 ImageNet weights ({e}); using random init")
            resnet = models.resnet50(weights=None)
        self.cnn_features = nn.Sequential(*list(resnet.children())[:-2])
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2048, nhead=8, dim_feedforward=2048, dropout=0.1),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        feats = self.cnn_features(x)  # (B, 2048, H, W)
        b, c, h, w = feats.shape
        seq = feats.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        seq = self.transformer_encoder(seq)
        feats = seq.permute(1, 2, 0).reshape(b, c, h, w)
        return self.classifier(feats)


MODEL_TYPES = {
    "baseline": {
        "factory": BaselineCNN,
        "label": "Baseline CNN",
        "label_ru": "Базовая CNN",
        "description": "Four-block convolutional network trained from scratch; the reference point.",
        "description_ru": "Свёрточная сеть из четырёх блоков, обученная с нуля; точка отсчёта.",
        "tagline": "reference",
        "tagline_ru": "эталон сравнения",
    },
    "efficientnet": {
        "factory": EfficientNetModel,
        "label": "EfficientNet-B0",
        "label_ru": "EfficientNet-B0",
        "description": "Compound-scaled network pre-trained on ImageNet; best accuracy/size balance.",
        "description_ru": "Сеть с компаундным масштабированием, предобучена на ImageNet; лучший баланс точности и размера.",
        "tagline": "recommended",
        "tagline_ru": "рекомендуется",
    },
    "mobilenet": {
        "factory": MobileNetModel,
        "label": "MobileNet-V2",
        "label_ru": "MobileNet-V2",
        "description": "Lightweight inverted-residual network; fastest inference, used for domain adaptation.",
        "description_ru": "Лёгкая сеть с инвертированными остаточными блоками; самая быстрая, используется в доменной адаптации.",
        "tagline": "fastest",
        "tagline_ru": "самая быстрая",
    },
    "hybrid": {
        "factory": HybridCNNTransformer,
        "label": "Hybrid CNN-Transformer",
        "label_ru": "Гибрид CNN-Transformer",
        "description": "ResNet-50 features refined by a Transformer encoder over the spatial grid.",
        "description_ru": "Признаки ResNet-50, уточнённые Transformer-энкодером по пространственной сетке.",
        "tagline": "largest",
        "tagline_ru": "самая крупная",
    },
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pick_device(spec: str | torch.device | None = "auto") -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    if spec and spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ModelManager:
    """Loads the classifiers, exposes predictions, metrics and the ensemble."""

    def __init__(self, models_dir="../models", device=None, trained_threshold: float = 0.5):
        self.models_dir = Path(models_dir)
        self.device = pick_device(device)
        self.trained_threshold = trained_threshold
        self.models: dict[str, dict] = {}
        self.class_names = self._load_class_names()
        self._metrics_cache: tuple[float, dict] | None = None
        print(f"Using device: {self.device}")

    # --- class names --------------------------------------------------------
    def _load_class_names(self) -> list[str]:
        class_file = self.models_dir / "class_names.json"
        if class_file.exists():
            with open(class_file, encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(f"{class_file} not found - train the models first (scripts/train_models.py)")

    # --- loading --------------------------------------------------------------
    def weights_path(self, model_name: str) -> Path | None:
        for candidate in (f"{model_name}_model.pth", f"{model_name}.pth", f"model_{model_name}.pth"):
            p = self.models_dir / candidate
            if p.exists():
                return p
        return None

    def load_model(self, model_name: str, model_type: str | None = None) -> nn.Module:
        model_type = model_type or model_name
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}")

        path = self.weights_path(model_name)
        # Only download ImageNet weights when we have nothing better.
        model = MODEL_TYPES[model_type]["factory"](num_classes=len(self.class_names), pretrained=path is None)
        loaded = False
        if path is not None:
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            loaded = True
            print(f"Loaded weights for {model_name} from {path.name}")
        else:
            print(f"No weights for {model_name} in {self.models_dir} - using untrained network")

        model = model.to(self.device).eval()
        self.models[model_name] = {"model": model, "type": model_type, "path": path, "weights_loaded": loaded}
        return model

    def get_model(self, model_name: str) -> nn.Module:
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' is not loaded. Loaded: {list(self.models)}")
        return self.models[model_name]["model"]

    def get_target_layer(self, model_name: str):
        """Last convolutional block, used by Grad-CAM."""
        entry = self.models[model_name]
        model, kind = entry["model"], entry["type"]
        if kind == "baseline":
            # The whole feature stack: its output is not modified in place (the inner ReLUs are).
            return model.features
        if kind in ("efficientnet", "mobilenet"):
            return model.backbone.features[-1]
        if kind == "hybrid":
            return model.cnn_features[-1]
        raise ValueError(kind)

    # --- metrics ----------------------------------------------------------------
    @property
    def metrics(self) -> dict:
        """models/model_metrics.json (written by scripts/evaluate_models.py), re-read when it changes."""
        path = self.models_dir / "model_metrics.json"
        if not path.exists():
            return {}
        mtime = path.stat().st_mtime
        if self._metrics_cache and self._metrics_cache[0] == mtime:
            return self._metrics_cache[1]
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._metrics_cache = (mtime, data)
        return data

    def is_trained(self, model_name: str) -> bool:
        m = self.metrics.get(model_name)
        if not m:
            return False
        return bool(m.get("trained", m.get("accuracy", 0) >= self.trained_threshold))

    def trained_models(self) -> list[str]:
        return [n for n in self.models if self.is_trained(n)]

    def best_model(self) -> str | None:
        trained = self.trained_models()
        if not trained:
            return next(iter(self.models), None)
        return max(trained, key=lambda n: self.metrics[n].get("accuracy", 0))

    def model_info(self, model_name: str) -> dict:
        entry = self.models.get(model_name, {})
        meta = MODEL_TYPES[entry.get("type", model_name)]
        m = self.metrics.get(model_name, {})
        model = entry.get("model")
        return {
            "name": model_name,
            "label": meta["label"],
            "label_ru": meta["label_ru"],
            "description": meta["description"],
            "description_ru": meta["description_ru"],
            "tagline": meta["tagline"],
            "tagline_ru": meta["tagline_ru"],
            "loaded": model is not None,
            "weights_loaded": bool(entry.get("weights_loaded")),
            "trained": self.is_trained(model_name),
            "parameters": m.get("parameters") or (f"{count_parameters(model)/1e6:.1f}M" if model else None),
            "size_mb": m.get("size_mb"),
            "accuracy": m.get("accuracy"),
            "f1_score": m.get("f1_score"),
            "inference_time_ms": m.get("inference_time_ms"),
            "evaluated_at": m.get("evaluated_at"),
        }

    def all_model_info(self) -> list[dict]:
        return [self.model_info(n) for n in self.models]

    # --- inference ----------------------------------------------------------------
    @torch.no_grad()
    def predict_probs(self, model_name: str, tensor: torch.Tensor) -> np.ndarray:
        out = self.get_model(model_name)(tensor.to(self.device))
        return F.softmax(out, dim=1)[0].float().cpu().numpy()

    def predict(self, model_name: str, tensor: torch.Tensor) -> dict:
        probs = self.predict_probs(model_name, tensor)
        idx = int(probs.argmax())
        return {
            "predicted_class": self.class_names[idx],
            "predicted_idx": idx,
            "confidence": float(probs[idx]),
            "probabilities": probs,
            "all_probabilities": probs.tolist(),
        }

    def compare_models(self, tensor: torch.Tensor, names=None) -> dict:
        names = names or list(self.models)
        return {n: self.predict(n, tensor) for n in names if n in self.models}

    def ensemble_weights(self, names) -> dict[str, float]:
        return {n: float(self.metrics.get(n, {}).get("accuracy", 1.0)) for n in names}

    def predict_ensemble(self, tensor: torch.Tensor, method: str = "weighted") -> dict:
        names = self.trained_models() or list(self.models)
        if not names:
            raise RuntimeError("No models loaded")
        probs = {n: self.predict_probs(n, tensor) for n in names}
        stack = np.stack([probs[n] for n in names])
        preds = [int(p.argmax()) for p in stack]

        if method == "weighted":
            w = np.array([self.ensemble_weights(names)[n] for n in names])
            ens = (stack * w[:, None]).sum(0) / w.sum()
        elif method == "average":
            ens = stack.mean(0)
        elif method == "voting":
            winner = Counter(preds).most_common(1)[0][0]
            ens = np.zeros(len(self.class_names))
            ens[winner] = 1.0
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        idx = int(ens.argmax())
        agreement = sum(p == idx for p in preds) / len(preds)
        entropy = float(-(ens * np.log(ens + 1e-10)).sum())
        max_entropy = float(np.log(len(self.class_names)))
        norm_entropy = entropy / max_entropy
        top_class = np.array([probs[n][idx] for n in names])
        std = float(top_class.std())
        mean = float(top_class.mean())

        if ens[idx] > 0.9 and agreement > 0.75 and norm_entropy < 0.3:
            level = "low"
        elif ens[idx] > 0.7 and agreement > 0.5 and norm_entropy < 0.6:
            level = "medium"
        else:
            level = "high"

        return {
            "predicted_class": self.class_names[idx],
            "predicted_idx": idx,
            "confidence": float(ens[idx]),
            "probabilities": ens,
            "all_probabilities": ens.tolist(),
            "ensemble_method": method,
            "models_used": names,
            "weights": self.ensemble_weights(names) if method == "weighted" else None,
            "individual_predictions": {
                n: {"predicted_class": self.class_names[preds[i]], "predicted_idx": preds[i], "confidence": float(stack[i].max())}
                for i, n in enumerate(names)
            },
            "agreement_rate": agreement,
            "uncertainty_metrics": {
                "prediction_variance": round(float(stack.var(0).mean()), 4),
                "entropy": round(entropy, 4),
                "normalized_entropy": round(norm_entropy, 4),
                "disagreement_score": round(1 - agreement, 4),
                "uncertainty_level": level,
                "confidence_interval": {
                    "lower": round(max(0.0, mean - 1.96 * std), 4),
                    "upper": round(min(1.0, mean + 1.96 * std), 4),
                    "width": round(min(1.0, mean + 1.96 * std) - max(0.0, mean - 1.96 * std), 4),
                },
            },
        }


class MaskedModel(nn.Module):
    """Wraps a classifier and hides every class except the allowed ones (used for the field model)."""

    def __init__(self, model: nn.Module, allowed_idx, num_classes: int):
        super().__init__()
        self.model = model
        mask = torch.full((num_classes,), float("-inf"))
        mask[list(allowed_idx)] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        return self.model(x) + self.mask


__all__ = [
    "BaselineCNN",
    "EfficientNetModel",
    "MobileNetModel",
    "HybridCNNTransformer",
    "MODEL_TYPES",
    "ModelManager",
    "MaskedModel",
    "count_parameters",
    "pick_device",
]
