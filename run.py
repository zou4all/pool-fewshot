#!/usr/bin/env python3
"""pool-fewshot — main entry point.

    python run.py selftest                      # no network, no model
    python run.py download --lat .. --lon ..    # fetch + tile + pre-annotate
    python run.py label                         # manual box editor
    python run.py check                         # dataset statistics
    python run.py ceiling                       # oracle upper bound
    python run.py experiment                    # sliding-window baseline

The dense method lives in run_dense.py, the full comparison in benchmark.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import annotate                                    # noqa: E402
from src.data import (Box, LAYER_RGB, cut_tiles, fetch_mosaic, iou,  # noqa: E402
                      load_labels, propose_pools, save_labels,
                      source_of, suggest_scales)
from src.geo import resolution                              # noqa: E402
from src.metrics import average_precision, counts, froc     # noqa: E402

TILES = ROOT / "data" / "tiles"
LABELS = ROOT / "data" / "labels"


# ===================================================================== self-test

def cmd_selftest(_args) -> int:
    """Check the maths without touching the network or a model."""
    from src.geo import Detection, deg2tile, pixel2deg, tile2deg, to_geojson

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK ' if cond else 'FAIL'}] {name} {detail}")

    print("\n-- geometry --")
    lat, lon, z = 43.518066, 5.461946, 19
    x, y = deg2tile(lat, lon, z)
    blat, blon = tile2deg(x, y, z)
    check("tile round-trip", abs(blat - lat) < 0.01 and abs(blon - lon) < 0.01,
          f"({x}, {y}) -> {blat:.4f}, {blon:.4f}")
    r = resolution(z, lat)
    check("resolution z19 ~0.22 m/px", 0.15 < r < 0.30, f"{r:.3f} m/px")
    la, lo = pixel2deg(x, y, z, 128, 128)
    check("pixel inside its tile", abs(la - blat) < 0.001)

    print("\n-- boxes --")
    a, b = Box(0, 0, 10, 10), Box(5, 5, 15, 15)
    check("IoU 10x10 / offset 5", abs(iou(a, b) - 25 / 175) < 1e-6,
          f"{iou(a, b):.4f}")
    check("IoU with itself = 1", abs(iou(a, a) - 1.0) < 1e-9)

    print("\n-- labels round-trip --")
    tmp = ROOT / "data" / "_selftest.txt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    save_labels(tmp, [Box(10, 20, 50, 60)], 256, 256)
    back = load_labels(tmp, 256, 256)
    check("read back", len(back) == 1 and abs(back[0].x1 - 10) < 0.5,
          f"{back[0].x1:.1f}, {back[0].y1:.1f}")
    tmp.unlink()

    print("\n-- tiling --")
    img = np.zeros((1000, 1000, 3), np.uint8)
    tiles = cut_tiles(img, 512, 64)
    check("tiles produced", len(tiles) >= 4, f"{len(tiles)} tiles")
    check("all 512x512", all(t[1].shape[:2] == (512, 512) for t in tiles))
    check("source_of cut tile", source_of("t_512_256") == "t")
    check("source_of plain name", source_of("pool_0042") == "pool_0042")

    print("\n-- metrics --")
    gt = [Box(10, 10, 50, 50)]
    perfect = [([(Box(10, 10, 50, 50), 0.9)], gt)]
    check("AP perfect = 1", abs(average_precision(perfect, 0.5) - 1.0) < 1e-6)
    wrong = [([(Box(200, 200, 240, 240), 0.9)], gt)]
    check("AP wrong = 0", average_precision(wrong, 0.5) < 1e-6)
    mixed = [([(Box(10, 10, 50, 50), 0.9), (Box(200, 200, 240, 240), 0.8)], gt)]
    check("AP 1 hit + 1 false alarm = 1",
          abs(average_precision(mixed, 0.5) - 1.0) < 1e-6,
          "(precision stays 1 up to full recall)")
    f = froc(mixed, 0.5)
    check("R@1fp = 1", abs(f["R@1fp"] - 1.0) < 1e-6)
    c = counts(mixed, 0.3)
    check("counts tp=1 fp=1", c["tp"] == 1 and c["fp"] == 1, str(c))

    print("\n-- colour proposals --")
    tile = np.full((256, 256, 3), (40, 90, 40), np.uint8)       # grass
    cv2.rectangle(tile, (100, 110), (140, 130), (200, 150, 60), -1)  # blue water
    props = propose_pools(tile, res_m=0.2)
    check("finds the blue rectangle", len(props) >= 1, f"{len(props)} proposal(s)")
    dark = np.full((256, 256, 3), (60, 45, 30), np.uint8)       # shadow
    check("ignores the shadow", len(propose_pools(dark, 0.2)) == 0)

    print("\n" + ("ALL GOOD" if ok else "SOMETHING IS BROKEN"))
    return 0 if ok else 1


# ===================================================================== download

def cmd_download(a) -> int:
    """Fetch an area, cut it into tiles, pre-annotate by colour."""
    span = a.span / 2
    bbox = (a.lat - span, a.lon - span, a.lat + span, a.lon + span)
    print(f"[dl] bbox {bbox}  zoom {a.zoom}")
    mosaic, origin = fetch_mosaic(bbox, a.zoom, LAYER_RGB,
                                  cache=str(ROOT / "data" / "raw"))
    print(f"[dl] mosaic {mosaic.shape[1]}x{mosaic.shape[0]} px, origin {origin}")

    res = resolution(a.zoom, a.lat)
    out_t = Path(a.out_tiles) if a.out_tiles else TILES
    out_l = Path(a.out_labels) if a.out_labels else LABELS
    out_t.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)

    prefix = a.prefix or f"{a.lat:.4f}_{a.lon:.4f}".replace(".", "p").replace("-", "m")
    kept = skipped = n_boxes = 0
    for name, crop, ox, oy in cut_tiles(mosaic, a.tile, a.overlap):
        if crop.mean() < 8:                       # tile outside coverage
            continue
        props = propose_pools(crop, res)

        if a.negatives_only:
            # Negatives: keep ONLY tiles where nothing is proposed, and write
            # no label file. Writing empty labels for tiles that do contain a
            # pool the colour filter missed would inject silent false
            # negatives into the ground truth.
            if props:
                skipped += 1
                continue
            cv2.imwrite(str(out_t / f"{prefix}_{name}.jpg"), crop)
            kept += 1
            continue

        if a.only_with_proposals and not props:
            skipped += 1
            continue
        cv2.imwrite(str(out_t / f"{prefix}_{name}.jpg"), crop)
        save_labels(out_l / f"{prefix}_{name}.txt", props, a.tile, a.tile)
        kept += 1
        n_boxes += len(props)

        if kept >= a.n_tiles > 0:
            break

    print(f"[dl] {kept} tiles written, {skipped} skipped, "
          f"{n_boxes} colour proposals")
    if not a.negatives_only:
        print("[dl] NEXT: run `python run.py label` and fix the boxes by hand.")
        print("     The colour filter only finds clean blue pools. Covered,")
        print("     green, shaded and empty pools MUST be added manually,")
        print("     otherwise you are benchmarking a colour threshold.")
    return 0


# ======================================================================== label

HELP_LABEL = """\
  drag left mouse : draw a box        d : delete the last box
  right click     : delete box under  c : clear the tile
  n / SPACE       : next tile         p : previous tile
  s               : save              q : save and quit
"""


def cmd_label(a) -> int:
    """Minimal box editor. No install, no account, works over any display."""
    tiles = sorted(Path(a.tiles).glob("*.jpg"))
    if not tiles:
        print(f"no tiles in {a.tiles}")
        return 1
    labels_dir = Path(a.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(HELP_LABEL)

    state = {"i": 0, "drag": None, "boxes": []}
    win = "pool-fewshot labeller"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def load(i):
        img = cv2.imread(str(tiles[i]))
        h, w = img.shape[:2]
        state["boxes"] = load_labels(labels_dir / (tiles[i].stem + ".txt"), w, h)
        return img

    def store(i, img):
        h, w = img.shape[:2]
        save_labels(labels_dir / (tiles[i].stem + ".txt"), state["boxes"], w, h)

    def on_mouse(event, x, y, _flags, _p):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            x0, y0 = state["drag"]
            state["drag"] = None
            if abs(x - x0) > 4 and abs(y - y0) > 4:
                state["boxes"].append(Box(min(x0, x), min(y0, y),
                                          max(x0, x), max(y0, y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            for b in list(state["boxes"]):
                if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2:
                    state["boxes"].remove(b)
                    break

    cv2.setMouseCallback(win, on_mouse)
    img = load(0)
    while True:
        view = annotate.draw_boxes(img, state["boxes"], colour=annotate.GREEN)
        view = annotate.title_bar(
            view, f"[{state['i'] + 1}/{len(tiles)}] {tiles[state['i']].name}"
                  f"  -  {len(state['boxes'])} box(es)")
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("n"), ord(" ")):
            store(state["i"], img)
            state["i"] = (state["i"] + 1) % len(tiles)
            img = load(state["i"])
        elif k == ord("p"):
            store(state["i"], img)
            state["i"] = (state["i"] - 1) % len(tiles)
            img = load(state["i"])
        elif k == ord("d") and state["boxes"]:
            state["boxes"].pop()
        elif k == ord("c"):
            state["boxes"] = []
        elif k == ord("s"):
            store(state["i"], img)
            print("saved")
        elif k == ord("q"):
            store(state["i"], img)
            break
    cv2.destroyAllWindows()
    return 0


# ======================================================================== check

def cmd_check(a) -> int:
    """Dataset statistics. Run this before any experiment."""
    tiles = sorted(Path(a.tiles).glob("*.jpg"))
    labels_dir = Path(a.labels)
    n_lab = n_box = 0
    sizes, per_tile, sources, dims = [], [], {}, {}
    for t in tiles:
        # Read the REAL image size rather than trusting --tile. YOLO labels are
        # normalised, so denormalising with the wrong side silently rescales
        # every object size reported below - and that number is what sets
        # --object-px, the most important setting in the whole pipeline.
        img = cv2.imread(str(t))
        if img is None:
            continue
        h, w = img.shape[:2]
        dims[(w, h)] = dims.get((w, h), 0) + 1
        boxes = load_labels(labels_dir / (t.stem + ".txt"), w, h)
        sources.setdefault(source_of(t.stem), []).append(t.stem)
        if boxes:
            n_lab += 1
            n_box += len(boxes)
            per_tile.append(len(boxes))
            sizes += [max(b.w(), b.h()) for b in boxes]

    print(f"tiles            : {len(tiles)}")
    print(f"annotated tiles  : {n_lab}")
    print(f"empty tiles      : {len(tiles) - n_lab}")
    print(f"objects          : {n_box}")
    print(f"source images    : {len(sources)}")
    shapes = sorted(dims.items(), key=lambda kv: -kv[1])
    print("tile size (px)   : " + ", ".join(f"{w}x{h} ({n})" for (w, h), n
                                            in shapes[:3]))
    if len(shapes) > 1:
        print("WARNING: tiles have mixed sizes. --object-px is a pixel count, "
              "so one value cannot fit every tile.")
    if sizes:
        s = np.array(sizes)
        print(f"object size (px) : median {np.median(s):.0f}  "
              f"p10 {np.percentile(s, 10):.0f}  p90 {np.percentile(s, 90):.0f}")
        print(f"objects / tile   : median {np.median(per_tile):.0f}  "
              f"max {max(per_tile)}")
        res = resolution(a.zoom, a.lat)
        print(f"object size (m)  : median {np.median(s) * res:.1f} m "
              f"(at zoom {a.zoom}, {res:.2f} m/px)")

        # Turn the statistics into the two settings that actually matter.
        from src.dense import PATCH_DEFAULT, target_size
        obj = float(np.median(s))
        (tw_, th_) = shapes[0][0]
        gh, gw = target_size((th_, tw_), obj, 4, PATCH_DEFAULT)
        grid = (gh // PATCH_DEFAULT) * (gw // PATCH_DEFAULT)
        need = int(max(per_tile)) * 16          # 4x4 squares per object
        pct = max(80.0, round(100.0 * (1 - need / max(1, grid)), 1))
        print(f"\nsuggested        : --object-px {obj:.0f} "
              f"--percentile {pct}")
        print(f"                   (grid {gh // PATCH_DEFAULT}x"
              f"{gw // PATCH_DEFAULT} = {grid} squares; the busiest tile has "
              f"{max(per_tile)} objects needing ~{need} of them)")
    if n_box < 50:
        print("\nWARNING: fewer than 50 objects. AP will be very noisy.")
    if len(sources) < 4:
        print("WARNING: fewer than 4 source images — the support/test split "
              "will be weak.")
    return 0


# ====================================================================== ceiling

def cmd_ceiling(a) -> int:
    """Oracle upper bound: what fraction of objects can the proposals reach?

    This separates the two failure modes. If the ceiling is low, no encoder can
    save the run — the candidate boxes simply never cover the objects. If the
    ceiling is high and AP is low, the encoder (or the score) is the problem.
    Diagnosing that distinction was what pointed to the proposal layer as the
    dominant factor.
    """
    tiles = sorted(Path(a.tiles).glob("*.jpg"))
    scales, ratios = suggest_scales(a.zoom, a.lat)
    print(f"[ceiling] scales {scales} ratios {ratios}")
    from src.sliding import sliding_windows

    hit = tot = 0
    best_ious = []
    for t in tiles:
        img = cv2.imread(str(t))
        gts = load_labels(Path(a.labels) / (t.stem + ".txt"),
                          img.shape[1], img.shape[0])
        if not gts:
            continue
        wins = sliding_windows(img.shape[:2], scales, ratios, a.stride)
        for g in gts:
            best = max((iou(w, g) for w in wins), default=0.0)
            best_ious.append(best)
            tot += 1
            hit += best >= a.iou
    if tot == 0:
        print("no annotated object")
        return 1
    print(f"[ceiling] {hit}/{tot} objects reachable at IoU>={a.iou} "
          f"= {100 * hit / tot:.1f}%")
    print(f"[ceiling] median best IoU {np.median(best_ious):.3f}")
    return 0


# =================================================================== experiment

def cmd_experiment(a) -> int:
    """Sliding-window baseline. Slow on purpose — it is the control."""
    import time

    from src.dense import make_encoder
    from src.fewshot import build_bank
    from src.sliding import detect_sliding

    tiles = sorted(Path(a.tiles).glob("*.jpg"))
    support_files = sorted(Path(a.support_dir).glob("*.jpg"))[:a.k_shot]
    if not support_files:
        print(f"no support crop in {a.support_dir}")
        return 1
    support = [cv2.imread(str(p)) for p in support_files]
    print(f"[exp] {len(support)} support crops, {len(tiles)} tiles")

    enc = make_encoder(a.model, fp16=not a.fp32,
                       trust_remote_code=a.trust_remote_code)
    bank = build_bank(enc, support)
    scales, ratios = suggest_scales(a.zoom, a.lat)

    per_image = []
    t0 = time.time()
    for i, t in enumerate(tiles):
        img = cv2.imread(str(t))
        gts = load_labels(Path(a.labels) / (t.stem + ".txt"),
                          img.shape[1], img.shape[0])
        boxes, scores = detect_sliding(enc, img, bank, scales, ratios,
                                       a.stride, max_boxes=a.max_boxes)
        per_image.append((list(zip(boxes, scores)), gts))
        print(f"[exp] {i + 1}/{len(tiles)}  {len(boxes)} boxes", end="\r")
    print(" " * 60, end="\r")

    print(f"\n[exp] {time.time() - t0:.0f} s")
    for thr in (0.5, 0.3):
        print(f"  AP@{thr} = {average_precision(per_image, thr):.3f}")
    for k, v in froc(per_image, 0.5).items():
        print(f"  {k} = {v:.3f}")
    print(f"  {counts(per_image, 0.3)}")
    return 0


# ========================================================================= main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="check the maths, no network").set_defaults(
        fn=cmd_selftest)

    d = sub.add_parser("download", help="fetch + tile + pre-annotate")
    d.add_argument("--lat", type=float, required=True)
    d.add_argument("--lon", type=float, required=True)
    d.add_argument("--zoom", type=int, default=19)
    d.add_argument("--span", type=float, default=0.01,
                   help="box side in degrees (0.01 ~ 1.1 km)")
    d.add_argument("--tile", type=int, default=512)
    d.add_argument("--overlap", type=int, default=64)
    d.add_argument("--n-tiles", type=int, default=0, help="0 = no limit")
    d.add_argument("--prefix", default="")
    d.add_argument("--out-tiles", default="")
    d.add_argument("--out-labels", default="")
    d.add_argument("--only-with-proposals", action="store_true")
    d.add_argument("--negatives-only", action="store_true",
                   help="keep only tiles with no proposal, write no labels")
    d.set_defaults(fn=cmd_download)

    la = sub.add_parser("label", help="manual box editor")
    la.add_argument("--tiles", default=str(TILES))
    la.add_argument("--labels", default=str(LABELS))
    la.set_defaults(fn=cmd_label)

    c = sub.add_parser("check", help="dataset statistics")
    c.add_argument("--tiles", default=str(TILES))
    c.add_argument("--labels", default=str(LABELS))
    c.add_argument("--tile", type=int, default=512)
    c.add_argument("--zoom", type=int, default=19)
    c.add_argument("--lat", type=float, default=43.5)
    c.set_defaults(fn=cmd_check)

    ce = sub.add_parser("ceiling", help="oracle upper bound of the proposals")
    ce.add_argument("--tiles", default=str(TILES))
    ce.add_argument("--labels", default=str(LABELS))
    ce.add_argument("--zoom", type=int, default=19)
    ce.add_argument("--lat", type=float, default=43.5)
    ce.add_argument("--stride", type=float, default=0.5)
    ce.add_argument("--iou", type=float, default=0.5)
    ce.set_defaults(fn=cmd_ceiling)

    e = sub.add_parser("experiment", help="sliding-window baseline (slow)")
    e.add_argument("--tiles", default=str(TILES))
    e.add_argument("--labels", default=str(LABELS))
    e.add_argument("--support-dir", default=str(ROOT / "data" / "support5"))
    e.add_argument("--k-shot", type=int, default=5)
    e.add_argument("--model", default="facebook/dinov2-base")
    e.add_argument("--zoom", type=int, default=19)
    e.add_argument("--lat", type=float, default=43.5)
    e.add_argument("--stride", type=float, default=0.5)
    e.add_argument("--max-boxes", type=int, default=200)
    e.add_argument("--fp32", action="store_true")
    e.add_argument("--trust-remote-code", action="store_true")
    e.set_defaults(fn=cmd_experiment)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
