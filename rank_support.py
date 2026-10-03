#!/usr/bin/env python3
"""Rank candidate support crops by their solo performance.

    python rank_support.py --half a --top 30 --out data/candidates

Support crops are not interchangeable. Measured on this dataset, the 5 best
crops score AP@0.3 = 0.554 while the 30 worst score 0.001 — a factor of 500
for the same model, the same tiles, the same code. Which crops you hand the
model is not a detail, it is the main lever.

LEAK WARNING. This script must run on a DIFFERENT half from the evaluation.
Ranking on the tiles you then report on means the crops were selected because
they score well there; the result is a self-fulfilling number. Rank on half
"a", evaluate with `run_dense.py --half b`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import Box, load_labels, split_half             # noqa: E402
from src.dense import (make_encoder, dense_prototype, detect,  # noqa: E402
                       patch_of, target_size)
from src.geo import resolution                                # noqa: E402
from src.metrics import average_precision                     # noqa: E402


def extract_candidates(tiles_dir: Path, labels_dir: Path, half: str,
                       margin: float = 0.25, min_px: int = 16):
    """Cut one crop per annotated object, with a margin of context.

    The margin matters: a pool cropped edge to edge is a rectangle of water
    with no coping, no terrace, no surroundings. Measured here, crop SIZE
    correlates +0.44 with final AP while sharpness correlates -0.38 — bigger
    crops carrying context beat tight, sharp ones.
    """
    out = []
    files = split_half(sorted(tiles_dir.glob("*.jpg")), half)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        for j, b in enumerate(load_labels(labels_dir / (f.stem + ".txt"), w, h)):
            mx, my = b.w() * margin, b.h() * margin
            x1, y1 = int(max(0, b.x1 - mx)), int(max(0, b.y1 - my))
            x2, y2 = int(min(w, b.x2 + mx)), int(min(h, b.y2 + my))
            if x2 - x1 < min_px or y2 - y1 < min_px:
                continue
            out.append((f"{f.stem}__{j}", img[y1:y2, x1:x2]))
    return out


def crop_stats(crop: np.ndarray) -> dict:
    """Cheap descriptors, useful for correlating "what makes a good crop"."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    b, _g2, r = (crop[:, :, i].astype(np.int16) for i in range(3))
    return {
        "w": crop.shape[1],
        "h": crop.shape[0],
        "area_px": int(crop.shape[0] * crop.shape[1]),
        "sharpness": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
        "median_v": int(np.median(hsv[:, :, 2])),
        "median_s": int(np.median(hsv[:, :, 1])),
        "blueness": int(np.median(b - r)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default=str(ROOT / "data" / "tiles"))
    p.add_argument("--labels", default=str(ROOT / "data" / "labels"))
    p.add_argument("--half", default="a", choices=["a", "b", "all"],
                   help="half used for ranking (NOT the evaluation half)")
    p.add_argument("--eval-tiles", type=int, default=24,
                   help="tiles used to score each candidate")
    p.add_argument("--model", default="facebook/dinov2-base")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--zoom", type=int, default=19)
    p.add_argument("--lat", type=float, default=43.5)
    p.add_argument("--object-m", type=float, default=6.0)
    p.add_argument("--patches-per-object", type=int, default=4)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--out", default=str(ROOT / "data" / "candidates"))
    p.add_argument("--csv", default=str(ROOT / "outputs" / "support_ranking.csv"))
    a = p.parse_args()

    tiles_dir, labels_dir = Path(a.tiles), Path(a.labels)
    cands = extract_candidates(tiles_dir, labels_dir, a.half, a.margin)
    if not cands:
        print(f"no candidate on half '{a.half}'")
        return 1
    print(f"[rank] {len(cands)} candidate crops on half '{a.half}'")

    # Scoring tiles: taken from the SAME half as the candidates, so the
    # evaluation half stays untouched.
    eval_files = split_half(sorted(tiles_dir.glob("*.jpg")), a.half)
    scored = []
    for f in eval_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gts = load_labels(labels_dir / (f.stem + ".txt"),
                          img.shape[1], img.shape[0])
        if gts:
            scored.append((f.stem, img, gts))
    scored = scored[:a.eval_tiles]
    print(f"[rank] scoring on {len(scored)} tiles, "
          f"{sum(len(g) for _, _, g in scored)} objects")

    enc = make_encoder(a.model, fp16=not a.fp32,
                       trust_remote_code=a.trust_remote_code)
    patch = patch_of(enc)
    object_px = a.object_m / resolution(a.zoom, a.lat)
    th, tw = target_size((512, 512), object_px, a.patches_per_object, patch)
    print(f"[rank] patch {patch}, tile resized to {tw}x{th}")

    rows = []
    for i, (name, crop) in enumerate(cands):
        bank = dense_prototype(enc, [crop], a.patches_per_object)
        per_image = []
        for _n, img, gts in scored:
            boxes, sc = detect(enc, img, bank, object_px,
                               patches_per_object=a.patches_per_object,
                               percentile=a.percentile)
            per_image.append(([(Box(*b), s) for b, s in zip(boxes, sc)], gts))
        ap5 = average_precision(per_image, 0.5)
        ap3 = average_precision(per_image, 0.3)
        rows.append({"name": name, "ap50": round(ap5, 4),
                     "ap30": round(ap3, 4), **crop_stats(crop)})
        print(f"[rank] {i + 1}/{len(cands)}  {name}  "
              f"AP@0.3={ap3:.3f}", end="\r")
    print(" " * 70, end="\r")

    rows.sort(key=lambda r: -r["ap30"])
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"[rank] ranking written to {a.csv}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lookup = dict(cands)
    for rank, r in enumerate(rows[:a.top], 1):
        cv2.imwrite(str(out / f"{rank:03d}_{r['ap30']:.3f}_{r['name']}.jpg"),
                    lookup[r["name"]])
    print(f"[rank] top {min(a.top, len(rows))} crops written to {out}")

    # What actually predicts a good crop? Correlations, not intuition.
    if len(rows) > 5:
        ap = np.array([r["ap30"] for r in rows])
        print("\n[rank] correlation with AP@0.3:")
        for key in ("area_px", "sharpness", "median_v", "median_s", "blueness"):
            v = np.array([r[key] for r in rows], dtype=float)
            if v.std() > 1e-9:
                print(f"         {key:<10} {np.corrcoef(v, ap)[0, 1]:+.2f}")

    print("\n[rank] best 8:")
    for r in rows[:8]:
        print(f"         AP@0.3 {r['ap30']:.3f}  {r['name']}  "
              f"{r['w']}x{r['h']}px")

    Path(a.csv).with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print("\n[rank] NEXT: python pick_support.py --n 5 "
          "--from data/candidates --to data/support5")
    print("       then  python run_dense.py --support-dir data/support5 --half b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
