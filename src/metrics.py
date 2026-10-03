"""Evaluation metrics and the episodic protocol."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .data import Box, iou


@dataclass
class Sample:
    name: str
    image: np.ndarray
    boxes: list = field(default_factory=list)


def average_precision(per_image, iou_thr: float = 0.5) -> float:
    """Global, COCO-style AP: every detection in the set ranked together.

    Do NOT average per-tile APs. A tile with no object necessarily returns 0,
    and averaging that in divides the score by the number of negatives — an
    artefact that measures nothing. In global AP an empty tile contributes
    through its false positives, which is exactly its role.
    """
    n_gt = sum(len(g) for _, g in per_image)
    if n_gt == 0:
        return 0.0
    flat = [(float(s), i, b) for i, (preds, _) in enumerate(per_image)
            for b, s in preds]
    if not flat:
        return 0.0
    flat.sort(key=lambda t: -t[0])

    matched = {i: [False] * len(g) for i, (_, g) in enumerate(per_image)}
    tp, fp = np.zeros(len(flat)), np.zeros(len(flat))
    for i, (_s, img, box) in enumerate(flat):
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
            tp[i] = 1
        else:
            fp[i] = 1

    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap, prev = 0.0, 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev)
        prev = r
    return float(ap)


def froc(per_image, iou_thr: float = 0.5, budgets=(1, 4, 16)) -> dict:
    """Recall reached for a budget of N false alarms PER TILE.

    When a detector produces many false positives, AP collapses towards zero
    and stops tracking progress. This curve stays informative in that regime,
    and answers the question a real user asks: "if I check N thumbnails per
    tile, how many pools do I actually see?"
    """
    n_gt = sum(len(g) for _, g in per_image)
    n_img = max(1, len(per_image))
    out = {f"R@{b}fp": 0.0 for b in budgets}
    if n_gt == 0:
        return out
    flat = [(float(s), i, b) for i, (preds, _) in enumerate(per_image)
            for b, s in preds]
    if not flat:
        return out
    flat.sort(key=lambda t: -t[0])

    matched = {i: [False] * len(g) for i, (_, g) in enumerate(per_image)}
    tp = fp = 0
    curve = []
    for _s, img, box in flat:
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
            tp += 1
        else:
            fp += 1
        curve.append((fp / n_img, tp / n_gt))

    for b in budgets:
        best = 0.0
        for fppi, r in curve:
            if fppi <= b:
                best = max(best, r)
            else:
                break
        out[f"R@{b}fp"] = float(best)
    return out


def counts(per_image, iou_thr: float = 0.3) -> dict:
    """Object-level true positives, false positives and misses.

    More readable than AP for a report: precision and recall counted in
    objects rather than as an area under a curve.
    """
    tp = fp = miss = 0
    for preds, gts in per_image:
        used = [False] * len(gts)
        order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
        for i in order:
            box = preds[i][0]
            best_j, best_v = -1, 0.0
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                v = iou(box, g)
                if v > best_v:
                    best_v, best_j = v, j
            if best_j >= 0 and best_v >= iou_thr:
                used[best_j] = True
                tp += 1
            else:
                fp += 1
        miss += used.count(False)
    return {"tp": tp, "fp": fp, "miss": miss,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + miss)}


def run_episodes(make_detector, support_pool, samples, k_shot: int = 5,
                 n_episodes: int = 10, iou_thrs=(0.5, 0.3), seed: int = 0,
                 verbose: bool = True) -> dict:
    """Repeat the experiment over n_episodes draws of k_shot support crops.

    The score depends heavily on which crops are drawn: a single number from a
    single draw is not informative. Mean AND standard deviation are reported.
    A large standard deviation is not noise to hide — it quantifies how
    sensitive the method is to the choice of examples, which is the practical
    question for anyone deploying it.
    """
    rng = random.Random(seed)
    res: dict = {}
    for ep in range(n_episodes):
        support = rng.sample(support_pool, min(k_shot, len(support_pool)))
        detector = make_detector(support)
        per_image = [(detector(s), s.boxes) for s in samples]
        for thr in iou_thrs:
            res.setdefault(f"AP@{thr}", []).append(
                average_precision(per_image, thr))
        for k, v in froc(per_image, iou_thrs[0]).items():
            res.setdefault(k, []).append(v)
        if verbose:
            line = "  ".join(f"{k}={res[k][-1]:.3f}"
                             for k in res if k.startswith("AP"))
            print(f"[eval] episode {ep + 1}/{n_episodes}  {line}")

    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "median": float(np.median(v)),
                "min": float(np.min(v)), "max": float(np.max(v)),
                "episodes": [float(x) for x in v]} for k, v in res.items()}


def print_summary(name: str, summary: dict) -> None:
    print(f"\n=== {name} ===")
    for k, s in summary.items():
        print(f"{k:>8} : {s['mean']:.3f} +/- {s['std']:.3f}"
              f"   (median {s['median']:.3f}, "
              f"min {s['min']:.3f} / max {s['max']:.3f})")
