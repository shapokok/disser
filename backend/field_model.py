"""
Field-condition model produced by the domain adaptation experiments.

The web app's "field images" mode routes predictions through a MobileNet-V2
that was adapted from PlantVillage (lab photos) to PlantDoc (field photos).
The adapted model only covers the classes that exist in both datasets, so
the prediction is restricted to that subset and the response says so.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

import config
from model import MobileNetModel


class FieldAdapter:
    def __init__(self, class_names, device, path=config.FIELD_MODEL_PATH, info_path=config.FIELD_MODEL_INFO):
        self.class_names = class_names
        self.device = device
        self.path = path
        self.model = None
        self.info = {}
        self.covered_idx: list[int] = []
        if info_path.exists():
            with open(info_path, encoding="utf-8") as f:
                self.info = json.load(f)
        self.covered_idx = [int(i) for i in self.info.get("covered_class_indices", [7, 9, 25, 30])]
        if path.exists():
            model = MobileNetModel(num_classes=len(class_names), pretrained=False)
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            self.model = model.to(device).eval()

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def covered_classes(self) -> list[str]:
        return [self.class_names[i] for i in self.covered_idx]

    @property
    def target_layer(self):
        return self.model.backbone.features[-1]

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Keep only the classes the adapted model was trained for."""
        mask = torch.full_like(logits, float("-inf"))
        mask[:, self.covered_idx] = 0
        return logits + mask

    @torch.no_grad()
    def predict_probs(self, tensor: torch.Tensor):
        logits = self.model(tensor.to(self.device))
        return F.softmax(self.mask_logits(logits), dim=1)[0].float().cpu().numpy()

    @torch.no_grad()
    def tta_probs(self, image) -> np.ndarray:
        """Test-time augmentation: the same 10 deterministic views as the DA experiments, averaged."""
        from explain import to_tensor

        small = image.resize(config.IMAGE_SIZE)
        views = [
            small,
            TF.hflip(small),
            TF.vflip(small),
            TF.vflip(TF.hflip(small)),
            TF.rotate(small, 10),
            TF.rotate(small, -10),
            TF.adjust_brightness(small, 1.2),
            TF.adjust_contrast(small, 1.2),
            TF.affine(small, angle=0, translate=[0, 0], scale=0.9, shear=[0.0]),
            TF.affine(small, angle=0, translate=[0, 0], scale=1.1, shear=[0.0]),
        ]
        batch = torch.cat([to_tensor(v) for v in views]).to(self.device)
        probs = F.softmax(self.mask_logits(self.model(batch)), dim=1)
        return probs.mean(0).float().cpu().numpy()

    def describe(self) -> dict:
        return {
            "available": self.available,
            "path": str(self.path),
            "covered_classes": self.covered_classes,
            "covered_class_indices": self.covered_idx,
            **{k: v for k, v in self.info.items() if k != "covered_class_indices"},
        }
