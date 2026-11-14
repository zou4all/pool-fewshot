#!/usr/bin/env python3
"""See what the model gets right and what it gets wrong.

    python show_results.py --support-dir data/support5 --half b --n 12
    python show_results.py --model facebook/dinov3-vitl16-pretrain-sat493m \\
                           --half b --heatmap

    green        correct detection (IoU >= threshold)
    red          false alarm
    blue dashed  missed object

Writes outputs/results/ with one annotated image per tile and a contact sheet.
A number tells you how well it works; this tells you WHY.
"""
from __future__ import annotations

import argparse
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
from src.data import Box, iou, load_labels, split_half          # noqa: E402
from src.dense import (make_encoder, dense_prototype, detect,   # noqa: E402
                       heatmap)
from src.geo import resolution                                  # noqa: E402


def match(dets, gts, thr: float):
    """Split detections into true/false positives and list the misses."""
    used = [False] * len(gts)
    tp, fp = [], []
    for box, score in sorted(dets, key=lambda d: -d[1]):
        best_j, best_v = -1, 0.0
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = iou(box, g)
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0 and best_v >= thr:
            used[best_j] = True
            tp.append((box, score))
        else:
            fp.append((box, score))
    missed = [g for g, u in zip(gts, used) if not u]
    return tp, fp, missed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default=str(ROOT / "data" / "tiles"))
    p.add_argument("--labels", default=str(ROOT / "data" / "labels"))
    p.add_argument("--support-dir", default=str(ROOT / "data" / "support5"))
    p.add_argument("--half", default="b", choices=["a", "b", "all"])
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
    p.add_argument("--n", type=int, default=12, help="number of tiles to draw")
    p.add_argument("--only-with-gt", action="store_true", default=True)
    p.add_argument("--include-empty", dest="only_with_gt",
                   action="store_false")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--object-m", type=float, default=6.0)
    p.add_argument("--zoom", type=int, default=19)
    p.add_argument("--lat", type=float, default=43.5)
    p.add_argument("--patches-per-object", type=int, default=4)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--max-boxes", type=int, default=30)
    p.add_argument("--min-patches", type=int, default=1,
                   help="smallest connected component that becomes a box; raise to 3-4 to kill speckle")
    p.add_argument("--agg", default="mean", choices=["mean", "knn"])
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--heatmap", action="store_true")
    p.add_argument("--out", default=str(ROOT / "outputs" / "results"))
    a = p.parse_args()

    sd = Path(a.support_dir)
    sf = (sorted(sd.glob("*.jpg")) + sorted(sd.glob("*.png")))[:a.k_shot]
    support = [c for c in (cv2.imread(str(f)) for f in sf) if c is not None]
    if not support:
        print(f"no support crop in {sd}")
        return 1

    enc = make_encoder(a.model, fp16=not a.fp32,
                       trust_remote_code=a.trust_remote_code,
                       load_in=_quant(a), layer=_layer(a), facet=a.facet)

    a.neg_bank = _neg_bank(a, enc)
    bank = dense_prototype(enc, support, a.patches_per_object)
    object_px = a.object_m / resolution(a.zoom, a.lat)

    files = split_half(sorted(Path(a.tiles).glob("*.jpg")), a.half)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    drawn, caps, n_tp, n_fp, n_miss = [], [], 0, 0, 0
    for f in files:
        if len(drawn) >= a.n:
            break
        img = cv2.imread(str(f))
        if img is None:
            continue
        gts = load_labels(Path(a.labels) / (f.stem + ".txt"),
                          img.shape[1], img.shape[0])
        if a.only_with_gt and not gts:
            continue
        boxes, scores = detect(enc, img, bank, object_px,
                               patches_per_object=a.patches_per_object,
                               percentile=a.percentile, max_boxes=a.max_boxes,
                               agg=a.agg, topk=a.topk,
                               min_patches=a.min_patches, decode=a.decode,
                               peak_floor=a.peak_floor,
                               peak_window=a.peak_window,
                               peak_rel=a.peak_rel,
                               max_side=a.max_side,
                               neg_bank=a.neg_bank, neg_weight=a.neg_weight)
        dets = [(Box(*b), s) for b, s in zip(boxes, scores)]
        tp, fp, missed = match(dets, gts, a.iou)
        n_tp, n_fp, n_miss = n_tp + len(tp), n_fp + len(fp), n_miss + len(missed)

        vis = annotate.draw_boxes(img, [b for b, _ in tp],
                                  [s for _, s in tp], annotate.GREEN)
        vis = annotate.draw_boxes(vis, [b for b, _ in fp],
                                  [s for _, s in fp], annotate.RED, thickness=1)
        vis = annotate.draw_boxes(vis, missed, colour=annotate.BLUE, dashed=True)
        cap = f"{f.stem}  tp {len(tp)} / fp {len(fp)} / miss {len(missed)}"
        cv2.imwrite(str(out_dir / f"{f.stem}_result.jpg"),
                    annotate.title_bar(vis, cap))

        if a.heatmap:
            cen, _raw = heatmap(enc, img, bank, object_px,
                                a.patches_per_object, a.topk, a.agg)
            cv2.imwrite(str(out_dir / f"{f.stem}_heat.jpg"),
                        annotate.heatmap_overlay(img, cen))

        drawn.append(vis)
        caps.append(cap)
        print(f"[show] {cap}")

    if not drawn:
        print("no tile matched the filter")
        return 1

    sheet = annotate.contact_sheet(drawn, cols=4, cell=320, captions=caps)
    sheet_path = out_dir / "sheet.jpg"
    annotate.save(sheet_path, sheet)

    prec = n_tp / max(1, n_tp + n_fp)
    rec = n_tp / max(1, n_tp + n_miss)
    print(f"\non these {len(drawn)} tiles, IoU >= {a.iou}:")
    print(f"  tp {n_tp}   fp {n_fp}   miss {n_miss}")
    print(f"  precision {prec:.2f}   recall {rec:.2f}")
    print(f"\nimages       -> {out_dir}")
    print(f"contact sheet-> {sheet_path}")
    print("\nlegend: green = correct, red = false alarm, "
          "blue dashed = missed object")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
