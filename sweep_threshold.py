#!/usr/bin/env python3
"""Choisir un seuil a partir de detections.csv, sans relancer le modele.

    python sweep_threshold.py
    python sweep_threshold.py outputs/bench/predictions/detections.csv

Le fichier contient deja chaque detection avec son score et son etiquette
(TP / FP / MISS). Balayer les seuils est donc instantane : aucun GPU, aucun
passage reseau. Le tableau montre ce que chaque seuil coute et rapporte.
"""
import csv
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/bench/predictions/detections.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))

tp = sorted(float(r["score"]) for r in rows if r["kind"] == "TP" and r["score"])
fp = sorted(float(r["score"]) for r in rows if r["kind"] == "FP" and r["score"])
n_miss = sum(1 for r in rows if r["kind"] == "MISS")
n_tiles = len({r["tile"] for r in rows})
n_gt = len(tp) + n_miss

if not tp:
    sys.exit("aucun TP dans ce fichier - mauvais chemin ?")

print(f"{path}")
print(f"{n_tiles} tuiles, {n_gt} objets reels, {len(tp)} trouves, "
      f"{len(fp)} fausses alertes, {n_miss} manques d'origine\n")
print(f"{'seuil':>7}{'gardes':>8}{'TP':>6}{'FP':>7}{'precision':>11}"
      f"{'rappel':>9}{'FP/tuile':>10}")
print("-" * 58)

lo, hi = min(tp + fp), max(tp + fp)
best = None
for i in range(21):
    t = lo + (hi - lo) * i / 20
    k_tp = sum(1 for s in tp if s >= t)
    k_fp = sum(1 for s in fp if s >= t)
    if k_tp + k_fp == 0:
        break
    prec = k_tp / (k_tp + k_fp)
    rec = k_tp / n_gt
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    mark = ""
    if best is None or f1 > best[0]:
        best, mark = (f1, t, prec, rec, k_tp, k_fp), " <-- meilleur F1"
    print(f"{t:>7.3f}{k_tp + k_fp:>8}{k_tp:>6}{k_fp:>7}{prec:>11.3f}"
          f"{rec:>9.3f}{k_fp / n_tiles:>10.1f}{mark}")

_f1, t, prec, rec, k_tp, k_fp = best
print(f"\nmeilleur compromis  : --min-score {t:.3f}")
print(f"                      precision {prec:.2f}, rappel {rec:.2f}, "
      f"{k_tp} trouvees, {k_fp} fausses alertes")
print("\nRappel : ce seuil est choisi SUR ce jeu. Pour une mesure honnete,")
print("le choisir sur la moitie 'a' et le verifier sur la moitie 'b'.")
