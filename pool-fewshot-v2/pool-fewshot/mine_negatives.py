#!/usr/bin/env python3
"""Mine hard negatives: the things that keep getting mistaken for the target.

    python mine_negatives.py --half a --top 40

Runs the current detector on tiles that contain NO annotated object, so every
box it returns is a false alarm by construction. The highest-scoring ones are
the hard negatives - on this dataset, bright roofs and terraces - and they are
saved as crops for `--neg-dir` to subtract at detection time.

This is the semi-supervised step of the project: no new labels are created, it
only uses the fact that some tiles are known to be empty.

LEAK WARNING, same rule as everywhere else. Mine on half "a", evaluate on
half "b". Mining on the half you report on means the negatives were chosen
*because* they score high there, and the gain you measure is your own
selection coming back to you.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import annotate                                         # noqa: E402
from src.data import load_labels, split_half                     # noqa: E402
from src.dense import dense_prototype, detect, make_encoder      # noqa: E402
from src.geo import resolution                                   # noqa: E402


def _quant(a):
    return "4bit" if a.load_4bit else ("8bit" if a.load_8bit else None)


def _layer(a):
    s = str(getattr(a, "layer", "") or "").strip()
    if not s:
        return None
    return float(s) if "." in s else int(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default=str(ROOT / "data" / "tiles"))
    p.add_argument("--labels", default=str(ROOT / "data" / "labels"))
    p.add_argument("--support-dir", default=str(ROOT / "data" / "support5"))
    p.add_argument("--half", default="a", choices=["a", "b", "all"],
                   help="mine here; evaluate on the OTHER half")
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-sat493m")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--load-8bit", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--layer", default="")
    p.add_argument("--max-side", type=int, default=1040)
    p.add_argument("--object-px", type=float, default=0.0)
    p.add_argument("--object-m", type=float, default=6.0)
    p.add_argument("--zoom", type=int, default=19)
    p.add_argument("--lat", type=float, default=43.5)
    p.add_argument("--patches-per-object", type=int, default=4)
    p.add_argument("--percentile", type=float, default=98.0)
    p.add_argument("--min-patches", type=int, default=3)
    p.add_argument("--per-tile", type=int, default=3,
                   help="keep at most this many per negative tile, so one busy "
                        "tile cannot dominate the bank")
    p.add_argument("--top", type=int, default=40, help="total crops to keep")
    p.add_argument("--out", default=str(ROOT / "data" / "negatives"))
    p.add_argument("--sheet", action="store_true")
    a = p.parse_args()

    labels_dir = Path(a.labels)
    files = split_half(sorted(Path(a.tiles).glob("*.jpg")), a.half)
    empty = [f for f in files if not (labels_dir / (f.stem + ".txt")).exists()]
    if not empty:
        print(f"no tile without labels on half '{a.half}'. Negatives are "
              "mined from tiles known to contain nothing - download some with "
              "`run.py download --negatives-only`.")
        return 1
    print(f"[mine] half '{a.half}': {len(files)} tiles, {len(empty)} of them empty")

    sd = Path(a.support_dir)
    sf = (sorted(sd.glob("*.jpg")) + sorted(sd.glob("*.png")))[:a.k_shot]
    support = [c for c in (cv2.imread(str(f)) for f in sf) if c is not None]
    if not support:
        print(f"no support crop in {sd}")
        return 1

    enc = make_encoder(a.model, fp16=not a.fp32,
                       trust_remote_code=a.trust_remote_code,
                       load_in=_quant(a), layer=_layer(a))
    bank = dense_prototype(enc, support, a.patches_per_object)
    object_px = a.object_px or a.object_m / resolution(a.zoom, a.lat)
    print(f"[mine] {enc.name}, object ~{object_px:.0f} px")

    found = []
    for i, f in enumerate(empty):
        img = cv2.imread(str(f))
        if img is None:
            continue
        boxes, scores = detect(enc, img, bank, object_px,
                               patches_per_object=a.patches_per_object,
                               percentile=a.percentile,
                               min_patches=a.min_patches,
                               max_side=a.max_side)
        order = sorted(range(len(boxes)), key=lambda j: -scores[j])
        for j in order[:a.per_tile]:
            x1, y1, x2, y2 = (int(round(v)) for v in boxes[j])
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size and crop.shape[0] > 8 and crop.shape[1] > 8:
                found.append((float(scores[j]), f.stem, crop))
        print(f"[mine] {i + 1}/{len(empty)}  {f.stem}", end="\r")
    print(" " * 60, end="\r")

    if not found:
        print("nothing mined - the detector returned no box on empty tiles")
        return 1

    found.sort(key=lambda t: -t[0])
    keep = found[:a.top]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for rank, (score, stem, crop) in enumerate(keep, 1):
        cv2.imwrite(str(out / f"{rank:03d}_{score:.3f}_{stem}.jpg"), crop)

    print(f"[mine] {len(found)} false alarms seen, {len(keep)} kept -> {out}")
    print(f"[mine] score range of the kept crops: "
          f"{keep[-1][0]:.3f} .. {keep[0][0]:.3f}")

    if a.sheet:
        imgs = [c for _s, _n, c in keep[:16]]
        caps = [f"{s:.3f} {n[:20]}" for s, n, _c in keep[:16]]
        sheet = annotate.contact_sheet(imgs, cols=4, cell=220, captions=caps)
        path = ROOT / "outputs" / "hard_negatives.jpg"
        annotate.save(path, sheet)
        print(f"[mine] contact sheet -> {path}")
        print("[mine] LOOK AT IT. If real pools appear there, your labels are "
              "incomplete and this bank would teach the model to ignore them.")

    print(f"\n[mine] NEXT: python run_dense.py --neg-dir {out} "
          f"--neg-weight 0.5 --half b ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
