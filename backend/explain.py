"""
Image preprocessing and explainable-AI visualisations (Grad-CAM, LIME).

All functions are device-aware and return small base64 JPEG previews instead
of large matplotlib figures, so API responses stay light.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config

_MEAN = np.array(config.IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(config.IMAGENET_STD, dtype=np.float32)


# --- basic I/O ---------------------------------------------------------------
def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_tensor(image: Image.Image, size=config.IMAGE_SIZE) -> torch.Tensor:
    """PIL image -> normalised float tensor (1, 3, H, W)."""
    arr = np.asarray(image.resize(size), dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


def preprocess_image(image_path, size=config.IMAGE_SIZE):
    """Backwards compatible helper: returns (tensor, original PIL image)."""
    image = load_image(image_path)
    return to_tensor(image, size), image


def encode_jpeg(image: Image.Image, max_side: int = config.PREVIEW_MAX_SIDE, quality: int = 85) -> str:
    """Downscale and encode a PIL image as a base64 JPEG string."""
    img = image.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@torch.no_grad()
def predict_probs(model, tensor: torch.Tensor, device) -> np.ndarray:
    out = model(tensor.to(device))
    return F.softmax(out, dim=1)[0].float().cpu().numpy()


# --- Grad-CAM ----------------------------------------------------------------
class GradCAM:
    """Minimal Grad-CAM (Selvaraju et al., 2017) for a chosen conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()

    def __call__(self, tensor: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = self.model(tensor)
            if class_idx is None:
                class_idx = int(output.argmax(dim=1).item())
            output[0, class_idx].backward()
        weights = self.gradients[0].mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * self.activations[0]).sum(0)
        cam = F.relu(cam).float().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = F.softmax(output.detach(), dim=1)[0].float().cpu().numpy()
        return cam, probs, class_idx


def _colorize(cam: np.ndarray, size) -> Image.Image:
    from matplotlib import colormaps

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    rgba = colormaps["jet"](np.asarray(cam_img, dtype=np.float32) / 255.0)
    return Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = config.GRADCAM_ALPHA):
    heat = _colorize(cam, image.size)
    return Image.blend(image, heat, alpha), heat


def explain_gradcam(model, target_layer, image_path, device, class_idx: int | None = None):
    """
    Returns predicted index, probabilities and preview images:
    {'original', 'heatmap', 'overlay'} (base64 JPEG).
    """
    image = load_image(image_path)
    tensor = to_tensor(image).to(device)
    gradcam = GradCAM(model, target_layer)
    try:
        cam, probs, idx = gradcam(tensor, class_idx)
    finally:
        gradcam.remove()
    overlay, heat = overlay_heatmap(image, cam)
    return {
        "predicted_idx": int(idx),
        "probabilities": probs,
        "images": {
            "original": encode_jpeg(image),
            "heatmap": encode_jpeg(heat),
            "overlay": encode_jpeg(overlay),
        },
    }


# --- LIME --------------------------------------------------------------------
def explain_lime(model, image_path, device, num_samples: int = config.LIME_NUM_SAMPLES, num_features: int = 8):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    image = load_image(image_path)
    arr = np.asarray(image.resize(config.IMAGE_SIZE), dtype=np.uint8)

    @torch.no_grad()
    def predict_fn(batch: np.ndarray) -> np.ndarray:
        x = (batch.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
        probs = []
        for chunk in x.split(64):
            probs.append(F.softmax(model(chunk), dim=1).float().cpu())
        return torch.cat(probs).numpy()

    explainer = lime_image.LimeImageExplainer(random_state=0)
    explanation = explainer.explain_instance(
        arr, predict_fn, top_labels=3, hide_color=0, num_samples=num_samples, random_seed=0
    )
    probs = predict_fn(arr[None])[0]
    idx = int(probs.argmax())
    temp, mask = explanation.get_image_and_mask(idx, positive_only=True, num_features=num_features, hide_rest=False)
    boundaries = (mark_boundaries(temp / 255.0, mask, color=(1, 1, 0)) * 255).astype(np.uint8)

    # Highlight the selected super-pixels, dim the rest.
    dim = (arr.astype(np.float32) * 0.35).astype(np.uint8)
    highlighted = np.where(mask[:, :, None] > 0, arr, dim)

    return {
        "predicted_idx": idx,
        "probabilities": probs,
        "images": {
            "original": encode_jpeg(image),
            "lime": encode_jpeg(Image.fromarray(boundaries)),
            "lime_regions": encode_jpeg(Image.fromarray(highlighted)),
        },
    }
