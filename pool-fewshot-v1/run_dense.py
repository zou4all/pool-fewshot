#!/usr/bin/env python3
"""Dense few-shot evaluation — the method that actually works.

    python run_dense.py --support-dir data/support5 --half b --k-shot 5

    python run_dense.py --model facebook/dinov3-vitl16-pretrain-sat493m \\
                        --support-dir data/support5 --half b --percentile 97

One forward pass per tile, a similarity heatmap, a percentile threshold,
connected components. ~60x faster than the sliding window for a much better
score.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import Box, load_labels, split_half                 # noqa: E402
from src.dense import make_encoder, dense_prototype, detect  # noqa: E402
from src.geo import resolution                                    # noqa: E402
from src.metrics import (average_precision, counts, froc,         # noqa: E402
                         run_episodes)


def load_support(folder: Path, k: int, seed: int = 0):
    """Load up to k support crops. Order is deterministic (sorted names)."""
    files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not files:
        raise SystemExit(f"no support crop in {folder}")
    if k > 0:
        files = files[:k]
    crops = [cv2.imread(str(p)) for p in files]
    return [c for c in crops if c is not None], files


def load_tiles(tiles_dir: Path, labels_dir: Path, half: str, limit: int = 0):
    """Load the evaluation tiles with their ground truth."""
    files = sorted(tiles_dir.glob("*.jpg"))
    files = split_half(files, half)
    if limit > 0:
        files = files[:limit]
    out = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gts = load_labels(labels_dir / (f.stem + ".txt"),
                          img.shape[1], img.shape[0])
        out.append((f.stem, img, gts))
    return out


def evaluate(encoder, bank, tiles, object_px, args, verbose=True):
    """Run detection over every tile and return (per_image, seconds)."""
    per_image = []
    t0 = time.time()
    for i, (name, img, gts) in enumerate(tiles):
        boxes, scores = detect(
            encoder, img, bank, object_px,
            patches_per_object=args.patches_per_object,
            percentile=args.percentile, max_boxes=args.max_boxes,
            pad=args.pad, min_score=args.min_score,
            agg=args.agg, topk=args.topk, min_patches=args.min_patches)
        dets = [(Box(*b), s) for b, s in zip(boxes, scores)]
        per_image.append((dets, gts))
        if verbose:
            print(f"[dense] {i + 1}/{len(tiles)}  {name}  "
                  f"{len(dets)} box(es)", end="\r")
    if verbose:
        print(" " * 70, end="\r")
    return per_image, time.time() - t0


def report(per_image, seconds, n_tiles) -> dict:
    res = {
        "AP@0.5": average_precision(per_image, 0.5),
        "AP@0.3": average_precision(per_image, 0.3),
        "seconds": round(seconds, 1),
        "sec_per_tile": round(seconds / max(1, n_tiles), 2),
        "n_tiles": n_tiles,
        "n_gt": sum(len(g) for _, g in per_image),
        "n_det": sum(len(d) for d, _ in per_image),
    }
    res.update(froc(per_image, 0.5))
    res.update({f"{k}@0.3": v for k, v in counts(per_image, 0.3).items()})
    return res


def print_report(name: str, r: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  AP@0.5        {r['AP@0.5']:.3f}")
    print(f"  AP@0.3        {r['AP@0.3']:.3f}")
    print(f"  R@1fp         {r.get('R@1fp', 0):.3f}")
    print(f"  R@4fp         {r.get('R@4fp', 0):.3f}")
    print(f"  R@16fp        {r.get('R@16fp', 0):.3f}")
    print(f"  precision@0.3 {r.get('precision@0.3', 0):.3f}"
          f"   recall@0.3 {r.get('recall@0.3', 0):.3f}")
    print(f"  tp {r.get('tp@0.3', 0)}  fp {r.get('fp@0.3', 0)}  "
          f"miss {r.get('miss@0.3', 0)}")
    print(f"  {r['n_tiles']} tiles, {r['n_gt']} objects, "
          f"{r['n_det']} detections, {r['seconds']} s "
          f"({r['sec_per_tile']} s/tile)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default=str(ROOT / "data" / "tiles"))
    p.add_argument("--labels", default=str(ROOT / "data" / "labels"))
    p.add_argument("--support-dir", default=str(ROOT / "data" / "support5"))
    p.add_argument("--half", default="b", choices=["a", "b", "all"],
                   help="evaluate on this half; rank support crops on the other")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--model", default="facebook/dinov2-base")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--fp32", action="store_true")

    p.add_argument("--object-px", type=float, default=0.0,
                   help="object size in tile pixels; 0 = derive from zoom")
    p.add_argument("--zoom", type=int, default=19)
    p.add_argument("--lat", type=float, default=43.5)
    p.add_argument("--object-m", type=float, default=6.0,
                   help="typical object side in metres")
    p.add_argument("--patches-per-object", type=int, default=4)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--max-boxes", type=int, default=30)
    p.add_argument("--min-patches", type=int, default=1,
                   help="smallest connected component that becomes a box; raise to 3-4 to kill speckle")
    p.add_argument("--pad", type=float, default=0.08)
    p.add_argument("--min-score", type=float, default=-1.0)
    p.add_argument("--agg", default="mean", choices=["mean", "knn"])
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--n-rot", type=int, default=4)

    p.add_argument("--episodes", type=int, default=0,
                   help=">0 : repeat with random draws and report mean +/- std")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    verbose = not a.quiet
    tiles = load_tiles(Path(a.tiles), Path(a.labels), a.half, a.limit)
    if not tiles:
        print("no tile to evaluate")
        return 1

    object_px = a.object_px or a.object_m / resolution(a.zoom, a.lat)
    if verbose:
        print(f"[dense] model {a.model}")
        print(f"[dense] half '{a.half}' -> {len(tiles)} tiles, "
              f"{sum(len(g) for _, _, g in tiles)} objects")
        print(f"[dense] object ~{object_px:.0f} px "
              f"({a.object_m} m at zoom {a.zoom})")

    enc = make_encoder(a.model, fp16=not a.fp32,
                       trust_remote_code=a.trust_remote_code)
    if verbose:
        print(f"[dense] patch {enc.patch}, dim {enc.dim}, device {enc.device}")

    # ---------------------------------------------------------- episodic mode
    if a.episodes > 0:
        pool_files = (sorted(Path(a.support_dir).glob("*.jpg"))
                      + sorted(Path(a.support_dir).glob("*.png")))
        pool = [cv2.imread(str(f)) for f in pool_files]
        pool = [c for c in pool if c is not None]

        class _S:
            def __init__(self, name, image, boxes):
                self.name, self.image, self.boxes = name, image, boxes

        samples = [_S(n, im, g) for n, im, g in tiles]

        def make_detector(support):
            bank = dense_prototype(enc, support, a.patches_per_object, a.n_rot)

            def run(sample):
                boxes, scores = detect(
                    enc, sample.image, bank, object_px,
                    patches_per_object=a.patches_per_object,
                    percentile=a.percentile, max_boxes=a.max_boxes,
                    pad=a.pad, min_score=a.min_score, agg=a.agg, topk=a.topk,
                    min_patches=a.min_patches)
                return [(Box(*b), s) for b, s in zip(boxes, scores)]
            return run

        summary = run_episodes(make_detector, pool, samples, a.k_shot,
                               a.episodes, seed=a.seed, verbose=verbose)
        print(f"\n=== {a.model} | {a.episodes} episodes, k={a.k_shot} ===")
        for k, s in summary.items():
            print(f"  {k:>8} : {s['mean']:.3f} +/- {s['std']:.3f}"
                  f"  (min {s['min']:.3f} / max {s['max']:.3f})")
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(
                {"model": a.model, "mode": "episodes", "k_shot": a.k_shot,
                 "episodes": a.episodes, "half": a.half,
                 "summary": summary}, indent=2))
            print(f"[dense] written to {a.out}")
        return 0

    # ------------------------------------------------------------- single run
    support, files = load_support(Path(a.support_dir), a.k_shot, a.seed)
    if verbose:
        print(f"[dense] {len(support)} support crops from {a.support_dir}")
        for f in files:
            print(f"          {f.name}")

    bank = dense_prototype(enc, support, a.patches_per_object, a.n_rot)
    if verbose:
        print(f"[dense] bank {bank.shape[0]} vectors "
              f"({len(support)} crops x {a.n_rot} rotations x 2 mirrors)")

    per_image, seconds = evaluate(enc, bank, tiles, object_px, a, verbose)
    r = report(per_image, seconds, len(tiles))
    r.update({"model": a.model, "support_dir": str(a.support_dir),
              "k_shot": len(support), "half": a.half,
              "percentile": a.percentile, "agg": a.agg,
              "patches_per_object": a.patches_per_object,
              "object_px": round(object_px, 1),
              "support_files": [f.name for f in files]})
    print_report(f"{a.model} | {Path(a.support_dir).name} | half {a.half}", r)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2))
        print(f"[dense] written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
