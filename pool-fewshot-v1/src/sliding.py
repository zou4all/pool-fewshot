"""Sliding-window baseline (crop-by-crop). Kept as a control.

This was the first approach: cut the tile into thousands of overlapping
windows, encode each one, compare it to the prototype. It works, but it is
~60x slower than the dense heatmap for a far worse score (AP@0.3 0.022 vs
0.554 on the same data). Keeping it in the repo is what lets the README claim
"the proposal layer is the dominant factor" with a measurement rather than an
opinion.
"""
from __future__ import annotations

import numpy as np

from .data import Box, nms


def sliding_windows(tile_hw, scales, ratios, stride_frac: float = 0.5):
    """Generate window boxes over a tile."""
    h, w = tile_hw
    out = []
    for s in scales:
        for r in ratios:
            bw, bh = int(round(s * (r ** 0.5))), int(round(s / (r ** 0.5)))
            if bw < 8 or bh < 8 or bw > w or bh > h:
                continue
            step_x = max(4, int(bw * stride_frac))
            step_y = max(4, int(bh * stride_frac))
            for y in range(0, h - bh + 1, step_y):
                for x in range(0, w - bw + 1, step_x):
                    out.append(Box(float(x), float(y),
                                   float(x + bw), float(y + bh), 0))
    return out


def detect_sliding(encoder, tile, bank, scales, ratios, stride_frac=0.5,
                   max_boxes=200, nms_thr=0.35, batch=64):
    """Score every window against the prototype bank, then NMS.

    max_boxes defaults to 200, not 50. With a few thousand windows per tile the
    single true positive can land around rank 145; truncating at 50 throws it
    away and prints AP = 0.000, which reads like a broken model rather than a
    truncated list.
    """
    h, w = tile.shape[:2]
    windows = sliding_windows((h, w), scales, ratios, stride_frac)
    if not windows:
        return [], []

    crops = [tile[int(b.y1):int(b.y2), int(b.x1):int(b.x2)] for b in windows]
    feats = []
    for i in range(0, len(crops), batch):
        feats.append(encoder.encode_crops(crops[i:i + batch]))
    feats = np.concatenate(feats, axis=0)

    sims = feats @ bank.T                 # (n_windows, n_prototypes)
    scores = sims.max(axis=1).tolist()    # best matching prototype wins

    order = np.argsort(scores)[::-1][:max_boxes * 3]
    boxes = [windows[i] for i in order]
    keep_s = [scores[i] for i in order]
    boxes, keep_s = nms(boxes, keep_s, nms_thr)
    return boxes[:max_boxes], keep_s[:max_boxes]
