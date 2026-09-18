#!/usr/bin/env python3
"""
Download the trained weights so the app runs on a machine that never trained anything.

    python scripts/download_weights.py                       # from the GitHub release (default tag)
    python scripts/download_weights.py --tag v2.1.0
    python scripts/download_weights.py --base-url https://example.org/weights/

The maintainer publishes the files once with:

    gh release create v2.0.0 models/baseline_model.pth models/efficientnet_model.pth \\
        models/mobilenet_model.pth models/field_mobilenet_da.pth models/model_metrics.json \\
        --title "Trained weights" --notes "PlantVillage classifiers + PlantDoc-adapted MobileNet-V2"

After downloading run `python scripts/evaluate_models.py` (or copy the published
model_metrics.json / results/metrics files) so the statistics page has numbers.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPO = "shapokok/disser"
FILES = [
    "baseline_model.pth",
    "efficientnet_model.pth",
    "mobilenet_model.pth",
    "hybrid_model.pth",
    "field_mobilenet_da.pth",
    "field_model_info.json",
    "model_metrics.json",
]


def download(url: str, dest: Path) -> bool:
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("content-length", 0))
            done = 0
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {dest.name}: {100 * done / total:5.1f} %", end="", flush=True)
            tmp.replace(dest)
            print(f"\r  {dest.name}: {done / 1e6:.1f} MB")
            return True
    except Exception as e:  # 404 for optional files is fine
        print(f"  {dest.name}: skipped ({e})")
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="v2.0.0", help="GitHub release tag")
    p.add_argument("--base-url", default=None, help="custom base URL instead of the GitHub release")
    p.add_argument("--files", nargs="+", default=FILES)
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args()
    base = args.base_url or f"https://github.com/{REPO}/releases/download/{args.tag}/"
    MODELS_DIR.mkdir(exist_ok=True)
    got = 0
    for name in args.files:
        dest = MODELS_DIR / name
        if dest.exists() and not args.force:
            print(f"  {name}: already present")
            got += 1
            continue
        got += download(base.rstrip("/") + "/" + name, dest)
    if not got:
        sys.exit("nothing downloaded - check the release tag or train the models (docs/TRAINING.md)")
    print(f"done: {got} file(s) in {MODELS_DIR}")


if __name__ == "__main__":
    main()
