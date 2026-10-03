#!/usr/bin/env python3
"""Import an external annotated dataset into this project. Two formats.

YOLO directory tree (images/<split>/ + labels/<split>/) - the usual Roboflow
export, and the format of the EcoAI pool dataset:

    python import_eco.py --yolo ../eco/EcoAI-CaseStudy/EcoAI-V1.0/dataset \\
                         --splits train --limit 120 --short --prefix "" --reset

COCO-style JSON:

    python import_eco.py --json path/testing.json --images path/images

TRAP WORTH KNOWING (COCO). In several public releases the `training.json` file
has NO `annotations` key - the geometry lives in `testing.json`. Reading the
wrong file gives you images with zero boxes and a benchmark that silently
measures nothing. This script checks for the key and says so.

TRAP WORTH KNOWING (YOLO). Labels are normalised, so they are only meaningful
together with the image they belong to. Every image is opened to read its real
size rather than assuming a square tile - a dataset with one odd-sized image
would otherwise corrupt that image's boxes with no error raised.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import Box, cut_tiles, load_labels, save_labels  # noqa: E402


RF_SUFFIX = re.compile(r"_(png|jpg|jpeg|tif|tiff|bmp)\.rf\.[0-9a-f]+$", re.I)
TRAILING_NUM = re.compile(r"^([A-Za-z]+).*[-_](\d+)$")


def clean_stem(stem: str, short: bool = False) -> str:
    """Roboflow file name -> usable tile name.

    'CANNES_TILES_512x512-1000_png.rf.8d2e5c...'  ->  'CANNES_TILES_512x512-1000'
    with --short                                  ->  'cannes-1000'

    The hash is not just ugly: `source_of()` splits on '_' and would read part
    of it as a tile offset. Short names also keep the labelling window legible.
    """
    stem = RF_SUFFIX.sub("", stem)
    if short:
        m = TRAILING_NUM.match(stem)
        if m:
            return f"{m.group(1).lower()}-{m.group(2)}"
    return stem


def import_yolo(a) -> int:
    """Import a YOLO tree: <root>/images/<split>/ + <root>/labels/<split>/."""
    root = Path(a.yolo)
    out_t, out_l = Path(a.out_tiles), Path(a.out_labels)
    if a.reset:
        for d in (out_t, out_l):
            for f in list(d.glob("*.jpg")) + list(d.glob("*.txt")):
                f.unlink()
        print(f"[import] reset: {out_t} and {out_l} emptied")
    out_t.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    pairs = []
    for split in splits:
        img_dir, lab_dir = root / "images" / split, root / "labels" / split
        if not img_dir.is_dir():
            img_dir, lab_dir = root / split / "images", root / split / "labels"
        if not img_dir.is_dir():
            print(f"[import] split '{split}' not found under {root}")
            continue
        found = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"[import] split '{split}': {len(found)} images")
        pairs += [(p, lab_dir / (p.stem + ".txt")) for p in found]

    if not pairs:
        print(f"[import] nothing found. Expected {root}/images/<split>/ "
              f"and {root}/labels/<split>/")
        return 1

    n_img = n_box = n_empty = 0
    sizes: dict = {}
    for img_path, lab_path in pairs:
        if a.limit and n_img >= a.limit:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        sizes[(w, h)] = sizes.get((w, h), 0) + 1

        boxes = load_labels(lab_path, w, h) if lab_path.exists() else []
        if not boxes:
            n_empty += 1
            if not a.keep_empty:
                continue

        stem = clean_stem(img_path.stem, a.short)
        if a.prefix:
            stem = f"{a.prefix}_{stem}"
        cv2.imwrite(str(out_t / f"{stem}.jpg"), img)
        if boxes:
            save_labels(out_l / f"{stem}.txt", boxes, w, h)
        n_img += 1
        n_box += len(boxes)

    print(f"\n[import] {n_img} tiles written, {n_box} boxes")
    if n_empty:
        kept = "kept" if a.keep_empty else "skipped"
        print(f"[import] {n_empty} images without a box ({kept})")
    shapes = sorted(sizes.items(), key=lambda kv: -kv[1])
    print("[import] tile sizes: " + ", ".join(f"{w}x{h} ({n})"
                                              for (w, h), n in shapes[:3]))
    if len(shapes) > 1:
        print("[import] NOTE: mixed sizes. Harmless here (every image is read "
              "at its own size), but --object-px is a pixel count, so one "
              "value cannot fit every tile perfectly.")
    print(f"[import] tiles  -> {out_t}")
    print(f"[import] labels -> {out_l}")
    print("[import] NEXT: python run.py check")
    return 0


def polygon_to_box(seg) -> tuple[float, float, float, float]:
    xs = seg[0::2]
    ys = seg[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yolo", default="",
                   help="root of a YOLO tree: <root>/images/<split>/ "
                        "and <root>/labels/<split>/")
    p.add_argument("--splits", default="train,val",
                   help="comma-separated splits to import (YOLO mode)")
    p.add_argument("--short", action="store_true",
                   help="CANNES_TILES_512x512-1000_png.rf.<hash> -> cannes-1000")
    p.add_argument("--keep-empty", action="store_true",
                   help="also import images with no box (as negatives)")
    p.add_argument("--reset", action="store_true",
                   help="empty the destination folders first")
    p.add_argument("--json", default="")
    p.add_argument("--images", default="")
    p.add_argument("--out-tiles", default=str(ROOT / "data" / "tiles"))
    p.add_argument("--out-labels", default=str(ROOT / "data" / "labels"))
    p.add_argument("--prefix", default="eco")
    p.add_argument("--tile", type=int, default=512,
                   help="0 = keep the source images whole")
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--category", default="",
                   help="keep only this category name")
    p.add_argument("--min-box", type=float, default=6.0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    if a.yolo:
        return import_yolo(a)
    if not a.json or not a.images:
        p.error("give --yolo <dir>, or both --json and --images")

    raw = json.loads(Path(a.json).read_text())
    if "annotations" not in raw:
        print(f"'{a.json}' has NO 'annotations' key.")
        print(f"keys present: {sorted(raw)}")
        print("In these datasets the geometry usually lives in testing.json, "
              "not training.json. Point --json at the file that has it.")
        return 1

    cats = {c["id"]: c.get("name", str(c["id"]))
            for c in raw.get("categories", [])}
    wanted = None
    if a.category:
        wanted = {i for i, n in cats.items() if n == a.category}
        if not wanted:
            print(f"category '{a.category}' not found. Available: "
                  f"{sorted(cats.values())}")
            return 1

    by_image = defaultdict(list)
    for ann in raw["annotations"]:
        if wanted and ann.get("category_id") not in wanted:
            continue
        if "bbox" in ann and ann["bbox"]:
            x, y, w, h = ann["bbox"]
            box = (x, y, x + w, y + h)
        elif ann.get("segmentation"):
            seg = ann["segmentation"]
            box = polygon_to_box(seg[0] if isinstance(seg[0], list) else seg)
        else:
            continue
        if box[2] - box[0] < a.min_box or box[3] - box[1] < a.min_box:
            continue
        by_image[ann["image_id"]].append(box)

    images = {im["id"]: im for im in raw.get("images", [])}
    print(f"[import] {len(raw['annotations'])} annotations, "
          f"{len(by_image)} images carrying at least one object")
    if cats:
        print(f"[import] categories: {sorted(cats.values())}")

    out_t, out_l = Path(a.out_tiles), Path(a.out_labels)
    out_t.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)
    img_dir = Path(a.images)

    n_img = n_tile = n_box = 0
    for img_id, boxes in by_image.items():
        meta = images.get(img_id, {})
        fname = meta.get("file_name")
        if not fname:
            continue
        path = img_dir / fname
        if not path.exists():
            hits = list(img_dir.rglob(Path(fname).name))
            if not hits:
                continue
            path = hits[0]
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = f"{a.prefix}_{Path(fname).stem}"
        n_img += 1

        if a.tile <= 0:
            cv2.imwrite(str(out_t / f"{stem}.jpg"), img)
            save_labels(out_l / f"{stem}.txt",
                        [Box(*b).clip(w, h) for b in boxes], w, h)
            n_tile += 1
            n_box += len(boxes)
        else:
            # Every tile keeps the SAME stem prefix, so source_of() groups them
            # and the support/test split cannot put one object on both sides.
            for name, crop, ox, oy in cut_tiles(img, a.tile, a.overlap):
                local = []
                for x1, y1, x2, y2 in boxes:
                    b = Box(x1 - ox, y1 - oy, x2 - ox, y2 - oy).clip(a.tile,
                                                                     a.tile)
                    if b.w() >= a.min_box and b.h() >= a.min_box:
                        local.append(b)
                if not local:
                    continue
                cv2.imwrite(str(out_t / f"{stem}_{name}.jpg"), crop)
                save_labels(out_l / f"{stem}_{name}.txt", local, a.tile, a.tile)
                n_tile += 1
                n_box += len(local)

        if a.limit and n_img >= a.limit:
            break

    print(f"[import] {n_img} source images -> {n_tile} tiles, {n_box} boxes")
    print(f"[import] tiles  -> {out_t}")
    print(f"[import] labels -> {out_l}")
    print("[import] NEXT: python run.py check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
