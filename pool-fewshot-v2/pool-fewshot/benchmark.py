#!/usr/bin/env python3
"""Full benchmark: compare every backbone, then run the best one.

    python benchmark.py                       # both phases, default models
    python benchmark.py --phase eval          # comparison only
    python benchmark.py --phase predict --best facebook/dinov2-base

PHASE 1 - EVALUATION
    Every model runs on the SAME tiles, the SAME support crops and the SAME
    settings. Only the backbone changes, so the difference between two rows is
    attributable to the backbone and to nothing else.
    -> outputs/bench/report.md, results.csv, results.json

PHASE 2 - PREDICTION
    The winner (highest AP@0.3, or --best) runs again and writes the images.
    -> outputs/bench/predictions/  + results_sheet.jpg

A gated model (DINOv3) that fails to load is reported as a skipped row rather
than crashing the run: a missing licence should not cost you the other five
measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

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
from src.dense import make_encoder, dense_prototype, detect  # noqa: E402
from src.geo import resolution                                  # noqa: E402
from src.metrics import average_precision, counts, froc         # noqa: E402

DEFAULT_MODELS = [
    "facebook/dinov2-base",
    "facebook/dinov2-with-registers-base",
    "facebook/dinov2-large",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-sat493m",
]

# Handy shortcuts for --models
ALIASES = {
    "dino": DEFAULT_MODELS,
    "dinov2": ["facebook/dinov2-base", "facebook/dinov2-with-registers-base",
               "facebook/dinov2-large"],
    "dinov3": ["facebook/dinov3-vitb16-pretrain-lvd1689m",
               "facebook/dinov3-vitl16-pretrain-lvd1689m",
               "facebook/dinov3-vitl16-pretrain-sat493m"],
    "siglip": ["google/siglip2-base-patch16-224"],
    "fast": ["facebook/dinov2-base",
             "facebook/dinov3-vitb16-pretrain-lvd1689m"],
    # v2 round: everything below loads with plain AutoModel and fits in 8 GB.
    "next": ["google/siglip2-base-patch16-224",
             "facebook/dinov2-with-registers-large",
             "facebook/dinov3-vits16-pretrain-lvd1689m",
             "facebook/dinov3-vith16plus-pretrain-lvd1689m"],
    # Does the size curve behave? 86M already beat 300M once.
    "sizes": ["facebook/dinov3-vits16-pretrain-lvd1689m",
              "facebook/dinov3-vitb16-pretrain-lvd1689m",
              "facebook/dinov3-vitl16-pretrain-lvd1689m",
              "facebook/dinov3-vith16plus-pretrain-lvd1689m"],
    # Do register tokens keep helping as the model grows?
    "registers": ["facebook/dinov2-base",
                  "facebook/dinov2-with-registers-base",
                  "facebook/dinov2-large",
                  "facebook/dinov2-with-registers-large"],
    # Needs --load-4bit on an 8 GB card: 6.7B params = 12.5 GB in fp16.
    # Run "quant-check" FIRST in both fp16 and 4bit - the gap between those
    # two numbers is the error bar on anything measured in "giants".
    "quant-check": ["facebook/dinov3-vitl16-pretrain-sat493m"],
    "giants": ["facebook/dinov3-vit7b16-pretrain-sat493m",
               "facebook/dinov3-vit7b16-pretrain-lvd1689m"],
}

NOTES = {
    "facebook/dinov2-base": "86M, web, patch 14 - the reference",
    "facebook/dinov2-with-registers-base": "86M, web, register tokens",
    "facebook/dinov2-large": "300M, web - watch for fp16 overflow",
    "facebook/dinov3-vitb16-pretrain-lvd1689m": "86M, web, patch 16",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "300M, web, patch 16",
    "facebook/dinov3-vitl16-pretrain-sat493m": "300M, SATELLITE pretraining",
    "google/siglip2-base-patch16-224": "86M, image-text, no CLS token",
    "facebook/dinov2-with-registers-large": "300M, web, registers - does the "
                                            "register gain scale?",
    "facebook/dinov3-vits16-pretrain-lvd1689m": "21M, web - the small end",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": "840M, web - the largest "
                                                    "that fits in 8 GB",
}

# NOT worth trying on this data, and why - so the next person does not repeat it:
#   Prithvi-EO (IBM/NASA)  6 spectral bands incl. NIR/SWIR, trained at 30 m/px.
#                          This imagery is RGB at 0.22 m/px - a 136x scale gap,
#                          and a pool is a fraction of one Prithvi pixel.
#                          Also needs TerraTorch, not AutoModel.
#   DINOv3 ViT-7B sat493m  the only other SAT-493M checkpoint, 6.7B parameters
#                          = 12.5 GB in fp16, ~27 GB to download (F32 on the
#                          Hub), and gated. Does not fit in 8 GB as-is, but
#                          --load-4bit brings the weights to ~3.1 GB. Loads
#                          with plain AutoModel, so no adapter is needed.
#   RemoteCLIP, GeoRSCLIP  RGB remote sensing and a genuinely good fit, but
#                          shipped as open_clip .pt checkpoints - needs an
#                          adapter before DenseEncoder can use them.


# ============================================================ shared machinery

def expand_models(spec: str) -> list[str]:
    if not spec:
        return list(DEFAULT_MODELS)
    out: list[str] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        out.extend(ALIASES.get(item, [item]))
    seen, uniq = set(), []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def load_dataset(tiles_dir: Path, labels_dir: Path, half: str, limit: int = 0):
    files = split_half(sorted(tiles_dir.glob("*.jpg")), half)
    if limit > 0:
        files = files[:limit]
    data = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gts = load_labels(labels_dir / (f.stem + ".txt"),
                          img.shape[1], img.shape[0])
        data.append((f.stem, img, gts))
    return data


def load_support(folder: Path, k: int):
    files = (sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")))[:k]
    crops = [cv2.imread(str(f)) for f in files]
    keep = [(f, c) for f, c in zip(files, crops) if c is not None]
    return [c for _f, c in keep], [f.name for f, _c in keep]


def run_model(model: str, dataset, support, object_px, a, verbose=True):
    """Load a model, detect on every tile, return (per_image, timings)."""
    t_load = time.time()
    enc = make_encoder(model, fp16=not a.fp32,
                       trust_remote_code=a.trust_remote_code,
                       load_in=_quant(a), layer=_layer(a), facet=a.facet)
    a.neg_bank = _neg_bank(a, enc)
    t_load = time.time() - t_load

    bank = dense_prototype(enc, support, a.patches_per_object, a.n_rot)
    per_image = []
    t0 = time.time()
    for i, (name, img, gts) in enumerate(dataset):
        boxes, scores = detect(enc, img, bank, object_px,
                               patches_per_object=a.patches_per_object,
                               percentile=a.percentile, max_boxes=a.max_boxes,
                               pad=a.pad, agg=a.agg, topk=a.topk,
                               min_patches=a.min_patches, decode=a.decode,
                               peak_floor=a.peak_floor,
                               peak_window=a.peak_window,
                               peak_rel=a.peak_rel,
                               max_side=a.max_side,
                               neg_bank=a.neg_bank, neg_weight=a.neg_weight)
        per_image.append(([(Box(*b), s) for b, s in zip(boxes, scores)], gts))
        if verbose:
            print(f"    {i + 1}/{len(dataset)}  {name}", end="\r")
    if verbose:
        print(" " * 60, end="\r")
    infer = time.time() - t0

    meta = {"patch": enc.patch, "dim": enc.dim, "device": enc.device,
            "fp16": enc.fp16, "load_s": round(t_load, 1)}
    del enc
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                    # noqa: BLE001
        pass
    return per_image, infer, meta


def score_row(model: str, per_image, infer: float, meta: dict,
              n_tiles: int) -> dict:
    row = {
        "model": model,
        "short": model.split("/")[-1],
        "AP@0.5": round(average_precision(per_image, 0.5), 4),
        "AP@0.3": round(average_precision(per_image, 0.3), 4),
    }
    row.update({k: round(v, 4) for k, v in froc(per_image, 0.5).items()})
    c = counts(per_image, 0.3)
    row.update({"tp": c["tp"], "fp": c["fp"], "miss": c["miss"],
                "precision": round(c["precision"], 4),
                "recall": round(c["recall"], 4)})
    row.update({"seconds": round(infer, 1),
                "sec_per_tile": round(infer / max(1, n_tiles), 2),
                "n_det": sum(len(d) for d, _ in per_image)})
    row.update(meta)
    row["note"] = NOTES.get(model, "")
    return row


def calibrate_threshold(per_image, fpr: float, iou_thr: float = 0.3) -> float:
    """Score above which the detector averages at most `fpr` false alarms/tile.

    A percentile threshold always returns boxes, including on empty tiles. For
    deployment you want an absolute cut-off. This walks the global ranking and
    stops at the requested false-positive budget.
    """
    n_img = max(1, len(per_image))
    flat = []
    for i, (dets, gts) in enumerate(per_image):
        for box, s in dets:
            flat.append((float(s), i, box))
    if not flat:
        return -1.0
    flat.sort(key=lambda t: -t[0])
    matched = {i: [False] * len(g) for i, (_, g) in enumerate(per_image)}
    fp = 0
    thr = flat[-1][0]
    for s, img, box in flat:
        gts = per_image[img][1]
        best_j, best_v = -1, 0.0
        for j, g in enumerate(gts):
            if matched[img][j]:
                continue
            v = iou(box, g)
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0 and best_v >= iou_thr:
            matched[img][best_j] = True
        else:
            fp += 1
            if fp / n_img > fpr:
                return float(s)
        thr = float(s)
    return thr


# ============================================================ phase 1: EVALUATE

def phase_eval(a) -> list[dict]:
    dataset = load_dataset(Path(a.tiles), Path(a.labels), a.half, a.limit)
    if not dataset:
        print("no tile: run `python run.py download` first")
        return []
    support, support_names = load_support(Path(a.support_dir), a.k_shot)
    if not support:
        print(f"no support crop in {a.support_dir}")
        return []

    object_px = a.object_px or a.object_m / resolution(a.zoom, a.lat)
    models = expand_models(a.models)
    n_gt = sum(len(g) for _, _, g in dataset)

    print("=" * 66)
    print("PHASE 1 - EVALUATION")
    print("=" * 66)
    print(f"  tiles     : {len(dataset)} (half '{a.half}'), {n_gt} objects")
    print(f"  support   : {len(support)} crops from {Path(a.support_dir).name}")
    for s in support_names:
        print(f"              {s}")
    print(f"  object    : ~{object_px:.0f} px ({a.object_m} m at zoom {a.zoom})")
    print(f"  percentile: {a.percentile}   agg: {a.agg}   "
          f"patches/object: {a.patches_per_object}")
    print(f"  models    : {len(models)}")
    print()

    rows, skipped = [], []
    for i, m in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {m}")
        try:
            per_image, infer, meta = run_model(m, dataset, support, object_px,
                                               a, a.verbose)
        except Exception as exc:                          # noqa: BLE001
            reason = str(exc).splitlines()[0][:150]
            print(f"    SKIPPED: {reason}")
            if a.verbose:
                traceback.print_exc()
            skipped.append({"model": m, "reason": reason})
            continue
        row = score_row(m, per_image, infer, meta, len(dataset))
        if a.calibrate:
            row["score_threshold"] = round(
                calibrate_threshold(per_image, a.fpr), 4)
        rows.append(row)
        print(f"    AP@0.5 {row['AP@0.5']:.3f}   AP@0.3 {row['AP@0.3']:.3f}"
              f"   R@4fp {row.get('R@4fp', 0):.3f}   {row['seconds']:.0f} s")

    if not rows:
        print("\nno model ran successfully")
        return []

    rows.sort(key=lambda r: -r["AP@0.3"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r})
        head = [k for k in ["short", "AP@0.3", "AP@0.5", "R@1fp", "R@4fp",
                            "R@16fp", "precision", "recall", "tp", "fp",
                            "miss", "sec_per_tile", "seconds", "patch", "dim",
                            "model", "note"] if k in keys]
        head += [k for k in keys if k not in head]
        wr = csv.DictWriter(fh, fieldnames=head)
        wr.writeheader()
        wr.writerows(rows)

    payload = {
        "config": {
            "half": a.half, "n_tiles": len(dataset), "n_gt": n_gt,
            "support_dir": str(a.support_dir), "k_shot": len(support),
            "support_files": support_names, "percentile": a.percentile,
            "agg": a.agg, "patches_per_object": a.patches_per_object,
            "object_px": round(object_px, 1), "zoom": a.zoom,
        },
        "results": rows,
        "skipped": skipped,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    (out / "report.md").write_text(write_report(rows, skipped, payload["config"]))

    print("\n" + "=" * 66)
    print("RANKING (by AP@0.3)")
    print("=" * 66)
    print(f"{'model':<42}{'AP@0.3':>8}{'AP@0.5':>8}{'R@4fp':>8}{'s/tile':>8}")
    for r in rows:
        print(f"{r['short']:<42}{r['AP@0.3']:>8.3f}{r['AP@0.5']:>8.3f}"
              f"{r.get('R@4fp', 0):>8.3f}{r['sec_per_tile']:>8.2f}")
    if skipped:
        print(f"\nskipped: {', '.join(s['model'] for s in skipped)}")
    print(f"\nreport -> {out / 'report.md'}")
    print(f"csv    -> {out / 'results.csv'}")
    return rows


def write_report(rows, skipped, cfg) -> str:
    L = ["# Backbone comparison — few-shot pool detection", ""]
    L += [f"- tiles: **{cfg['n_tiles']}** (half `{cfg['half']}`), "
          f"**{cfg['n_gt']}** annotated objects",
          f"- support: **{cfg['k_shot']} crops** from `{cfg['support_dir']}`",
          f"- settings: percentile {cfg['percentile']}, agg `{cfg['agg']}`, "
          f"{cfg['patches_per_object']} patches/object, "
          f"object ~{cfg['object_px']:.0f} px",
          "", "Only the backbone changes between rows. Everything else is "
          "identical, so the gap between two rows is attributable to the "
          "backbone.", ""]
    L += ["## Results", "",
          "| model | AP@0.3 | AP@0.5 | R@1fp | R@4fp | precision | recall "
          "| s/tile | notes |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['short']}` | **{r['AP@0.3']:.3f}** | {r['AP@0.5']:.3f} "
                 f"| {r.get('R@1fp', 0):.3f} | {r.get('R@4fp', 0):.3f} "
                 f"| {r['precision']:.2f} | {r['recall']:.2f} "
                 f"| {r['sec_per_tile']:.2f} | {r['note']} |")
    if skipped:
        L += ["", "## Skipped models", ""]
        for s in skipped:
            L.append(f"- `{s['model']}` — {s['reason']}")
        L += ["", "DINOv3 weights are gated: accept the licence on the model "
              "page, then `hf auth login`."]

    if rows:
        best, worst = rows[0], rows[-1]
        L += ["", "## Reading the table", "",
              f"- Best: **{best['short']}**, AP@0.3 {best['AP@0.3']:.3f}.",
              f"- Spread between best and worst: "
              f"{best['AP@0.3'] - worst['AP@0.3']:+.3f} AP@0.3 — same code, "
              f"same crops, same tiles.",
              "- `R@4fp` is the operational figure: the share of objects found "
              "while allowing 4 false alarms per tile. AP alone collapses "
              "towards zero in a high-false-positive regime and stops "
              "discriminating.",
              "- `precision` / `recall` are counted at IoU 0.3, which is the "
              "right threshold when the aim is *flagging a location* rather "
              "than segmenting it precisely."]
    L += ["", "---", "Imagery © IGN — BD ORTHO®, Open Licence / Etalab."]
    return "\n".join(L) + "\n"


# ============================================================= phase 2: PREDICT

def phase_predict(a, rows=None) -> int:
    out = Path(a.out)
    best = a.best
    threshold = a.min_score
    if not best:
        src = rows
        if src is None and (out / "results.json").exists():
            src = json.loads((out / "results.json").read_text())["results"]
        if not src:
            print("no model to run: give --best, or run the eval phase first")
            return 1
        best = max(src, key=lambda r: r["AP@0.3"])["model"]
        if a.calibrate:
            hit = [r for r in src if r["model"] == best]
            if hit and "score_threshold" in hit[0]:
                threshold = hit[0]["score_threshold"]

    dataset = load_dataset(Path(a.tiles), Path(a.labels), a.half, a.limit)
    support, support_names = load_support(Path(a.support_dir), a.k_shot)
    if not dataset or not support:
        print("missing tiles or support crops")
        return 1
    object_px = a.object_px or a.object_m / resolution(a.zoom, a.lat)

    print("\n" + "=" * 66)
    print("PHASE 2 - PREDICTION")
    print("=" * 66)
    print(f"  model     : {best}")
    print(f"  tiles     : {len(dataset)}")
    if threshold > -1.0:
        print(f"  threshold : {threshold:.4f} "
              f"(calibrated for <= {a.fpr} false alarms/tile)")

    enc = make_encoder(best, fp16=not a.fp32,
                       trust_remote_code=a.trust_remote_code,
                       load_in=_quant(a), layer=_layer(a), facet=a.facet)

    a.neg_bank = _neg_bank(a, enc)
    bank = dense_prototype(enc, support, a.patches_per_object, a.n_rot)

    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows_csv, gallery, caps = [], [], []
    tot_tp = tot_fp = tot_miss = 0
    t0 = time.time()

    for i, (name, img, gts) in enumerate(dataset):
        boxes, scores = detect(enc, img, bank, object_px,
                               patches_per_object=a.patches_per_object,
                               percentile=a.percentile, max_boxes=a.max_boxes,
                               pad=a.pad, min_score=threshold,
                               agg=a.agg, topk=a.topk,
                               min_patches=a.min_patches, decode=a.decode,
                               peak_floor=a.peak_floor,
                               peak_window=a.peak_window,
                               peak_rel=a.peak_rel,
                               max_side=a.max_side,
                               neg_bank=a.neg_bank, neg_weight=a.neg_weight)
        dets = [(Box(*b), s) for b, s in zip(boxes, scores)]

        used = [False] * len(gts)
        tp, fp = [], []
        for box, s in sorted(dets, key=lambda d: -d[1]):
            best_j, best_v = -1, 0.0
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                v = iou(box, g)
                if v > best_v:
                    best_v, best_j = v, j
            if best_j >= 0 and best_v >= a.iou:
                used[best_j] = True
                tp.append((box, s))
            else:
                fp.append((box, s))
        missed = [g for g, u in zip(gts, used) if not u]
        tot_tp, tot_fp = tot_tp + len(tp), tot_fp + len(fp)
        tot_miss += len(missed)

        vis = annotate.draw_boxes(img, [b for b, _ in tp],
                                  [s for _, s in tp], annotate.GREEN)
        vis = annotate.draw_boxes(vis, [b for b, _ in fp],
                                  [s for _, s in fp], annotate.RED, thickness=1)
        vis = annotate.draw_boxes(vis, missed, colour=annotate.BLUE, dashed=True)
        cap = f"{name}  tp{len(tp)} fp{len(fp)} miss{len(missed)}"
        cv2.imwrite(str(pred_dir / f"{name}_pred.jpg"),
                    annotate.title_bar(vis, cap))

        if len(gallery) < a.sheet and (gts or a.sheet_neg):
            gallery.append(vis)
            caps.append(cap)

        res_m = resolution(a.zoom, a.lat)
        for (box, s), kind in ([(d, "TP") for d in tp] +
                               [(d, "FP") for d in fp]):
            rows_csv.append({"tile": name, "kind": kind,
                             "x1": round(box.x1, 1), "y1": round(box.y1, 1),
                             "x2": round(box.x2, 1), "y2": round(box.y2, 1),
                             "score": round(float(s), 4),
                             "area_m2": round(box.area() * res_m * res_m, 1)})
        for g in missed:
            rows_csv.append({"tile": name, "kind": "MISS",
                             "x1": round(g.x1, 1), "y1": round(g.y1, 1),
                             "x2": round(g.x2, 1), "y2": round(g.y2, 1),
                             "score": "", "area_m2":
                                 round(g.area() * res_m * res_m, 1)})
        print(f"    {i + 1}/{len(dataset)}  {cap}", end="\r")
    print(" " * 70, end="\r")
    secs = time.time() - t0

    with open(pred_dir / "detections.csv", "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["tile", "kind", "x1", "y1", "x2",
                                            "y2", "score", "area_m2"])
        wr.writeheader()
        wr.writerows(rows_csv)

    if gallery:
        sheet = annotate.contact_sheet(gallery, cols=4, cell=320, captions=caps)
        annotate.save(out / "results_sheet.jpg", sheet)

    prec = tot_tp / max(1, tot_tp + tot_fp)
    rec = tot_tp / max(1, tot_tp + tot_miss)
    summary = {"model": best, "half": a.half, "n_tiles": len(dataset),
               "iou": a.iou, "threshold": threshold,
               "tp": tot_tp, "fp": tot_fp, "miss": tot_miss,
               "precision": round(prec, 4), "recall": round(rec, 4),
               "seconds": round(secs, 1),
               "support_files": support_names}
    (out / "predictions_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  tp {tot_tp}   fp {tot_fp}   miss {tot_miss}")
    print(f"  precision {prec:.2f}   recall {rec:.2f}   (IoU {a.iou})")
    print(f"  {secs:.0f} s for {len(dataset)} tiles")
    print(f"\n  images  -> {pred_dir}")
    print(f"  table   -> {pred_dir / 'detections.csv'}")
    if gallery:
        print(f"  sheet   -> {out / 'results_sheet.jpg'}")
    print("\n  legend: green = correct, red = false alarm, "
          "blue dashed = missed object")
    return 0


# ========================================================================= main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", default="all", choices=["all", "eval", "predict"])
    p.add_argument("--models", default="",
                   help="comma-separated; aliases: "
                        + ", ".join(sorted(ALIASES)))
    p.add_argument("--best", default="",
                   help="force the model used by the prediction phase")

    p.add_argument("--tiles", default=str(ROOT / "data" / "tiles"))
    p.add_argument("--labels", default=str(ROOT / "data" / "labels"))
    p.add_argument("--support-dir", default=str(ROOT / "data" / "support5"))
    p.add_argument("--half", default="b", choices=["a", "b", "all"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--k-shot", type=int, default=5)

    p.add_argument("--object-px", type=float, default=0.0)
    p.add_argument("--object-m", type=float, default=6.0)
    p.add_argument("--zoom", type=int, default=19)
    p.add_argument("--lat", type=float, default=43.5)
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
    p.add_argument("--iou", type=float, default=0.3)

    p.add_argument("--calibrate", action="store_true",
                   help="derive an absolute score threshold from --fpr")
    p.add_argument("--fpr", type=float, default=4.0,
                   help="false-alarm budget per tile used by --calibrate")

    p.add_argument("--sheet", type=int, default=12,
                   help="tiles shown on the contact sheet")
    p.add_argument("--sheet-neg", action="store_true",
                   help="also include tiles with no ground truth")
    p.add_argument("--out", default=str(ROOT / "outputs" / "bench"))
    p.add_argument("--fp32", action="store_true")
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
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    rows = None
    if a.phase in ("all", "eval"):
        rows = phase_eval(a)
        if a.phase == "eval":
            return 0 if rows else 1
        if not rows:
            return 1
    return phase_predict(a, rows)


if __name__ == "__main__":
    raise SystemExit(main())
