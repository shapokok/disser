#!/usr/bin/env python3
"""
Export a trained classifier to ONNX (for mobile / edge deployment) and verify
that ONNX Runtime reproduces the PyTorch outputs.

    uv pip install onnx onnxscript onnxruntime          # optional dependencies
    python scripts/export_onnx.py --model mobilenet     # -> models/mobilenet_model.onnx
    python scripts/export_onnx.py --model mobilenet --checkpoint models/field_mobilenet_da.pth --out models/field_mobilenet_da.onnx

The exported graph takes a float32 tensor [N, 3, 224, 224] normalised with the
ImageNet mean/std (see backend/explain.py) and returns raw logits [N, 38].
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model import MODEL_TYPES

MODELS_DIR = PROJECT_ROOT / "models"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mobilenet", choices=list(MODEL_TYPES))
    p.add_argument("--checkpoint", default=None, help="state dict (default: models/<model>_model.pth)")
    p.add_argument("--out", default=None, help="output .onnx (default: models/<model>_model.onnx)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else MODELS_DIR / f"{args.model}_model.pth"
    out = Path(args.out) if args.out else MODELS_DIR / f"{args.model}_model.onnx"
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    model = MODEL_TYPES[args.model]["factory"](num_classes=38, pretrained=False)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state) if isinstance(state, dict) else state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB)")

    if args.no_verify:
        return
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed - skipping verification (uv pip install onnxruntime)")
        return
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        ref = model(x).numpy()
    t0 = time.time()
    got = sess.run(None, {"image": x.numpy()})[0]
    dt = (time.time() - t0) / 4 * 1000
    err = float(np.abs(ref - got).max())
    print(f"max |pytorch - onnxruntime| = {err:.2e}   onnxruntime {dt:.1f} ms/img (CPU)")
    if err > 1e-3:
        sys.exit("verification failed: outputs differ")


if __name__ == "__main__":
    main()
