#!/usr/bin/env python3
"""Copy the top-N ranked crops into a support folder, and show them.

    python pick_support.py --n 5 --from data/candidates --to data/support5
    python pick_support.py --n 5 --from data/candidates --to data/support5 --sheet

Crops written by rank_support.py are named `<rank>_<ap>_<tile>__<index>.jpg`,
so plain alphabetical order is already ranked order.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import annotate                                        # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="src", default=str(ROOT / "data" / "candidates"))
    p.add_argument("--to", dest="dst", default=str(ROOT / "data" / "support5"))
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--skip", type=int, default=0,
                   help="skip the first N (to build a 'worst crops' control)")
    p.add_argument("--reverse", action="store_true",
                   help="take the WORST instead of the best (control run)")
    p.add_argument("--sheet", action="store_true",
                   help="also write a contact sheet of the chosen crops")
    p.add_argument("--clear", action="store_true",
                   help="empty the destination folder first")
    a = p.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    files = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    if not files:
        print(f"no crop in {src}")
        return 1
    if a.reverse:
        files = files[::-1]
    chosen = files[a.skip:a.skip + a.n]
    if not chosen:
        print("selection is empty (check --skip / --n)")
        return 1

    dst.mkdir(parents=True, exist_ok=True)
    if a.clear:
        for old in list(dst.glob("*.jpg")) + list(dst.glob("*.png")):
            old.unlink()

    for i, f in enumerate(chosen, 1):
        shutil.copy2(f, dst / f"{i:02d}_{f.name}")
        print(f"  {i:02d}  {f.name}")
    print(f"\n{len(chosen)} crops -> {dst}")

    if a.sheet:
        imgs, caps = [], []
        for f in chosen:
            img = cv2.imread(str(f))
            if img is not None:
                imgs.append(img)
                caps.append(f.stem[:38])
        sheet = annotate.contact_sheet(imgs, cols=min(5, max(1, len(imgs))),
                                       cell=200, captions=caps)
        out = ROOT / "outputs" / f"support_{dst.name}.jpg"
        annotate.save(out, sheet)
        print(f"contact sheet -> {out}")

    print(f"\nNEXT: python run_dense.py --support-dir {dst} --half b "
          f"--k-shot {len(chosen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
