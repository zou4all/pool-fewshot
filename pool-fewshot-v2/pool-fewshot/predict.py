#!/usr/bin/env python3
"""Inference on new images — input in, annotated output out.

    python predict.py --images my_folder/ --support-dir data/support5
    python predict.py --lat 43.518 --lon 5.462 --zoom 19 --span 0.006
    python predict.py --images tile.jpg --heatmap --geojson out.geojson

Writes to outputs/predictions/:
    <name>_pred.jpg      annotated image
    <name>_heat.jpg      similarity map (with --heatmap)
    detections.csv       one row per box
    detections.geojson   WGS84 coordinates (only for --lat/--lon input)

No ground truth needed. This is the deployment path.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _quant(a):
    """CLI flags -> the load_in value DenseEncoder expects."""
    return "4bit" if a.load_4bit else ("8bit" if a.load_8bit else None)


def _layer(a):
    """--layer text -> int, float or None."""
    s = str(getattr(a, "layer", "") or "").strip()
    if not s:
        return None
    return float(s) if "." in s else int(s)


def _neg_bank(a, enc):
    """Load the hard-negative crops and encode them, or None."""
    folder = str(getattr(a, "neg_dir", "") or "").strip()
    if not folder or getattr(a, "neg_weight", 0) == 0:
        return None
    import cv2
    from src.dense import dense_prototype
    files = sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.png"))
    crops = [c for c in (cv2.imread(str(f)) for f in files) if c is not None]
    if not crops:
        print(f"[warn] no crop in {folder}, running without negatives")
        return None
    bank = dense_prototype(enc, crops, a.patches_per_object, n_rot=1)
    print(f"[neg] {len(crops)} hard negatives -> bank of {bank.shape[0]} "
          f"vectors, weight {a.neg_weight}")
    return bank

from src import annotate                                        # noqa: E402
from src.data import LAYER_RGB, cut_tiles, fetch_mosaic         # noqa: E402
from src.dense import (make_encoder, dense_prototype, detect,   # noqa: E402
                       heatmap)
from src.geo import Detection, resolution, save_geojson, to_geojson  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def gather_images(path: Path):
    """One file or a whole folder -> [(name, image)]."""
    if path.is_file():
        img = cv2.imread(str(path))
        return [(path.stem, img)] if img is not None else []
    out = []
    for f in sorted(path.iterdir()):
        if f.suffix.lower() in IMG_EXT:
            img = cv2.imread(str(f))
            if img is not None:
                out.append((f.stem, img))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input (choose one)")
    src.add_argument("--images", default="", help="an image or a folder")
    src.add_argument("--lat", type=float, default=None)
    src.add_argument("--lon", type=float, default=None)
    src.add_argument("--span", type=float, default=0.006)
    src.add_argument("--zoom", type=int, default=19)
    src.add_argument("--tile", type=int, default=512)
    src.add_argument("--overlap", type=int, default=64)

    p.add_argument("--support-dir", default=str(ROOT / "data" / "support5"))
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--model", default="facebook/dinov2-base")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--load-4bit", action="store_true",
                   help="NF4 quantisation: fits a model ~4x bigger than your VRAM")
    p.add_argument("--load-8bit", action="store_true",
                   help="int8 quantisation: half the VRAM, less feature drift than 4bit")
    p.add_argument("--neg-dir", default="",
                   help="folder of hard-negative crops (see mine_negatives.py); their similarity is subtracted from the score")
    p.add_argument("--neg-weight", type=float, default=1.0,
                   help="how hard to subtract the negatives; 0 disables, 0.5 is a good start")
    p.add_argument("--max-side", type=int, default=1040,
                   help="cap on the resized tile; lower it to fit a big model in VRAM, but lower it for EVERY model in a comparison")
    p.add_argument("--decode", default="percentile",
                   choices=["percentile", "peaks"],
                   help="how to turn the heatmap into boxes; 'peaks' has no per-tile budget")
    p.add_argument("--peak-floor", type=float, default=90.0,
                   help="--decode peaks: percentile below which peaks are ignored")
    p.add_argument("--peak-window", type=int, default=0,
                   help="--decode peaks: neighbourhood for the local max; 0 derives it from --patches-per-object")
    p.add_argument("--peak-rel", type=float, default=0.6,
                   help="--decode peaks: how far down from the peak a box grows; lower = bigger boxes")
    p.add_argument("--facet", default="token",
                   choices=["token", "key", "query", "value"],
                   help="which part of the block to use as the descriptor; keys are the least sensitive to background clutter")
    p.add_argument("--layer", default="",
                   help="read features from an intermediate block: an int (1=first, -1=last) or a fraction (0.75). Empty = the final output.")
    p.add_argument("--fp32", action="store_true")

    p.add_argument("--object-m", type=float, default=6.0)
    p.add_argument("--lat-ref", type=float, default=43.5,
                   help="latitude used for the metres/pixel conversion")
    p.add_argument("--patches-per-object", type=int, default=4)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--max-boxes", type=int, default=30)
    p.add_argument("--min-patches", type=int, default=1,
                   help="smallest connected component that becomes a box; raise to 3-4 to kill speckle")
    p.add_argument("--min-score", type=float, default=-1.0)
    p.add_argument("--agg", default="mean", choices=["mean", "knn"])
    p.add_argument("--topk", type=int, default=3)

    p.add_argument("--out", default=str(ROOT / "outputs" / "predictions"))
    p.add_argument("--heatmap", action="store_true")
    p.add_argument("--sheet", action="store_true")
    p.add_argument("--geojson", default="")
    a = p.parse_args()

    # ---------------------------------------------------------------- support
    sd = Path(a.support_dir)
    sf = (sorted(sd.glob("*.jpg")) + sorted(sd.glob("*.png")))[:a.k_shot]
    if not sf:
        print(f"no support crop in {sd}")
        return 1
    support = [cv2.imread(str(f)) for f in sf]
    support = [c for c in support if c is not None]
    print(f"[predict] {len(support)} support crops from {sd.name}")

    enc = make_encoder(a.model, fp16=not a.fp32,
                       trust_remote_code=a.trust_remote_code,
                       load_in=_quant(a), layer=_layer(a), facet=a.facet)

    a.neg_bank = _neg_bank(a, enc)
    bank = dense_prototype(enc, support, a.patches_per_object)
    print(f"[predict] {enc.name}, patch {enc.patch}, device {enc.device}")

    # ------------------------------------------------------------------ input
    origin = None
    if a.lat is not None and a.lon is not None:
        s = a.span / 2
        mosaic, origin = fetch_mosaic((a.lat - s, a.lon - s, a.lat + s, a.lon + s),
                                      a.zoom, LAYER_RGB,
                                      cache=str(ROOT / "data" / "raw"))
        images = [(n, c) for n, c, _ox, _oy in
                  cut_tiles(mosaic, a.tile, a.overlap)]
        offsets = {n: (ox, oy) for n, _c, ox, oy in
                   cut_tiles(mosaic, a.tile, a.overlap)}
        a.lat_ref = a.lat
    elif a.images:
        images = gather_images(Path(a.images))
        offsets = {}
    else:
        print("give --images, or --lat and --lon")
        return 1
    if not images:
        print("no readable image")
        return 1

    object_px = a.object_m / resolution(a.zoom, a.lat_ref)
    print(f"[predict] {len(images)} image(s), object ~{object_px:.0f} px")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, geo_dets, sheet_imgs, sheet_caps = [], [], [], []

    for i, (name, img) in enumerate(images):
        boxes, scores = detect(enc, img, bank, object_px,
                               patches_per_object=a.patches_per_object,
                               percentile=a.percentile, max_boxes=a.max_boxes,
                               min_score=a.min_score, agg=a.agg, topk=a.topk,
                               min_patches=a.min_patches, decode=a.decode,
                               peak_floor=a.peak_floor,
                               peak_window=a.peak_window,
                               peak_rel=a.peak_rel,
                               max_side=a.max_side,
                               neg_bank=a.neg_bank, neg_weight=a.neg_weight)
        vis = annotate.draw_boxes(img, boxes, scores, colour=annotate.GREEN)
        vis = annotate.title_bar(vis, f"{name} - {len(boxes)} detection(s)")
        cv2.imwrite(str(out_dir / f"{name}_pred.jpg"), vis)

        if a.sheet and len(sheet_imgs) < 16:
            sheet_imgs.append(vis)
            sheet_caps.append(f"{name} ({len(boxes)})")

        if a.heatmap:
            cen, _raw = heatmap(enc, img, bank, object_px,
                                a.patches_per_object, a.topk, a.agg)
            cv2.imwrite(str(out_dir / f"{name}_heat.jpg"),
                        annotate.heatmap_overlay(img, cen))

        ox, oy = offsets.get(name, (0, 0))
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            rows.append({"image": name, "x1": round(x1, 1), "y1": round(y1, 1),
                         "x2": round(x2, 1), "y2": round(y2, 1),
                         "score": round(float(s), 4),
                         "area_m2": round((x2 - x1) * (y2 - y1)
                                          * resolution(a.zoom, a.lat_ref) ** 2, 1)})
            if origin is not None:
                geo_dets.append(Detection(x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                                          float(s)))
        print(f"[predict] {i + 1}/{len(images)}  {name}  "
              f"{len(boxes)} box(es)", end="\r")
    print(" " * 60, end="\r")

    csv_path = out_dir / "detections.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "x1", "y1", "x2", "y2",
                                            "score", "area_m2"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n[predict] {len(rows)} detections over {len(images)} image(s)")
    print(f"[predict] annotated images -> {out_dir}")
    print(f"[predict] table            -> {csv_path}")

    if a.sheet and sheet_imgs:
        sheet = annotate.contact_sheet(sheet_imgs, cols=4, cell=320,
                                       captions=sheet_caps)
        annotate.save(out_dir / "sheet.jpg", sheet)
        print(f"[predict] contact sheet    -> {out_dir / 'sheet.jpg'}")

    if origin is not None:
        gj_path = a.geojson or str(out_dir / "detections.geojson")
        save_geojson(gj_path, to_geojson(geo_dets, origin, a.zoom))
        print(f"[predict] GeoJSON          -> {gj_path}  (opens in QGIS)")
    elif a.geojson:
        print("[predict] GeoJSON needs --lat/--lon: a plain image carries "
              "no geographic reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
