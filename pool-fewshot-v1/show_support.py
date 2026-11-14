#!/usr/bin/env python3
"""Planche contact des vignettes de support deja choisies.

    python show_support.py
    python show_support.py data/support20

Pas besoin de data/candidates : ce script lit directement le dossier de
support et ecrit une planche dans outputs/.
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import annotate                                        # noqa: E402

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/support5")
files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
if not files:
    raise SystemExit(f"aucune vignette dans {folder}")

imgs, caps = [], []
for f in files:
    img = cv2.imread(str(f))
    if img is not None:
        h, w = img.shape[:2]
        imgs.append(img)
        caps.append(f"{f.name}  {w}x{h}")
        print(f"  {f.name:<28} {w:>4} x {h:<4} px")

out = ROOT / "outputs" / f"support_{folder.name}.jpg"
annotate.save(out, annotate.contact_sheet(imgs, cols=min(5, len(imgs)),
                                          cell=220, captions=caps))
print(f"\n{len(imgs)} vignettes -> {out}")
