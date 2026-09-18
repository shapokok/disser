#!/usr/bin/env python3
"""
Find (near-)duplicate images between the PlantVillage train and valid splits.

PlantVillage is known to contain repeated and augmented copies of the same
photo, so a random split leaks information from train into valid and inflates
accuracy. This script hashes every image (64-bit difference hash) and reports
valid images whose hash is identical or within a small Hamming distance of a
train image.

    python scripts/check_duplicates.py               # all 88k images, ~5 min on 4 cores
    python scripts/check_duplicates.py --distance 6  # looser near-duplicate threshold

Outputs:
    results/metrics/duplicates.json   per-class counts + list of leaked valid images
    results/metrics/duplicates.md     human readable summary
`scripts/evaluate_models.py` reads duplicates.json and adds "accuracy on the
de-duplicated valid subset" to the metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
OUT_DIR = PROJECT_ROOT / "results" / "metrics"
EXT = {".jpg", ".jpeg", ".png"}


def dhash(path: str, size: int = 8) -> int:
    """Difference hash: resize to (size+1)x size grayscale, compare horizontal neighbours."""
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return -1
    px = list(img.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def list_split(split: str) -> list[tuple[str, str]]:
    root = DATA_DIR / split
    items = []
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cls.iterdir()):
            if f.suffix.lower() in EXT:
                items.append((cls.name, str(f)))
    return items


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance", type=int, default=4, help="max Hamming distance for a near-duplicate (0 = exact)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    train, valid = list_split("train"), list_split("valid")
    print(f"hashing {len(train)} train + {len(valid)} valid images with {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        train_h = pool.map(dhash, [f for _, f in train], chunksize=256)
        valid_h = pool.map(dhash, [f for _, f in valid], chunksize=256)

    # exact matches via dict; near matches via bucketed search on 4 x 16-bit slices (pigeonhole: any pair within
    # distance <= 3 shares at least one identical slice; for distance 4-7 we widen with a second pass on 8 slices)
    by_hash: dict[int, list[int]] = defaultdict(list)
    for i, h in enumerate(train_h):
        by_hash[h].append(i)
    n_slices = 4 if args.distance <= 3 else 8
    width = 64 // n_slices
    buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(n_slices)]
    for i, h in enumerate(train_h):
        for s in range(n_slices):
            buckets[s][(h >> (s * width)) & ((1 << width) - 1)].append(i)

    leaked = []
    exact = 0
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_leaked: dict[str, int] = defaultdict(int)
    cross_class = 0
    for j, (cls, f) in enumerate(valid):
        h = valid_h[j]
        per_class_total[cls] += 1
        if h < 0:
            continue
        best = None
        if h in by_hash:
            best = (0, by_hash[h][0])
        else:
            seen = set()
            for s in range(n_slices):
                for i in buckets[s].get((h >> (s * width)) & ((1 << width) - 1), ()):
                    if i in seen:
                        continue
                    seen.add(i)
                    d = hamming(h, train_h[i])
                    if d <= args.distance and (best is None or d < best[0]):
                        best = (d, i)
        if best is not None:
            d, i = best
            exact += d == 0
            per_class_leaked[cls] += 1
            if train[i][0] != cls:
                cross_class += 1
            leaked.append(
                {
                    "valid": str(Path(f).relative_to(DATA_DIR)),
                    "train": str(Path(train[i][1]).relative_to(DATA_DIR)),
                    "distance": d,
                    "same_class": train[i][0] == cls,
                }
            )
        if j % 2000 == 0:
            print(f"  {j}/{len(valid)}  leaked so far {len(leaked)}", flush=True)

    per_class = {
        c: {
            "valid": per_class_total[c],
            "leaked": per_class_leaked[c],
            "share": per_class_leaked[c] / per_class_total[c],
        }
        for c in sorted(per_class_total)
    }
    summary = {
        "distance": args.distance,
        "train_images": len(train),
        "valid_images": len(valid),
        "valid_with_near_duplicate_in_train": len(leaked),
        "share": len(leaked) / len(valid),
        "exact_duplicates": exact,
        "cross_class_duplicates": cross_class,
        "per_class": per_class,
        "leaked": leaked,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "duplicates.json").write_text(json.dumps(summary, indent=1))
    worst = sorted(per_class.items(), key=lambda kv: -kv[1]["share"])[:8]
    md = [
        "# PlantVillage train/valid leakage check",
        "",
        f"Hash: 64-bit dHash, near-duplicate threshold Hamming ≤ {args.distance}.",
        "",
        f"* valid images with a near-duplicate in train: **{len(leaked)} of {len(valid)} ({100 * len(leaked) / len(valid):.1f} %)**",
        f"* exact hash matches: {exact}; duplicates labelled with a different class: {cross_class}",
        "",
        "| Class | valid | leaked | share |",
        "|---|---|---|---|",
    ] + [f"| {c} | {v['valid']} | {v['leaked']} | {100 * v['share']:.1f} % |" for c, v in worst]
    (OUT_DIR / "duplicates.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:6]))
    print(f"wrote {OUT_DIR / 'duplicates.json'}")


if __name__ == "__main__":
    sys.exit(main())
