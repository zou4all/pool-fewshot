"""Crop-level prototypes: CL2N centring, cosine scoring.

Used by the sliding-window baseline. The dense path builds its prototypes in
dense.py, but the underlying idea is the same one: a class is represented by
the mean of its examples, and an unknown is assigned by cosine similarity
(Snell et al., "Prototypical Networks", 2017).
"""
from __future__ import annotations

import numpy as np

from .dense import RawPixelEncoder, l2  # noqa: F401  (re-export)


def centre_l2(feats: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """CL2N: subtract the mean, then re-normalise (SimpleShot, Wang 2019).

    In high dimension a few "hub" vectors sit close to everything and win every
    comparison regardless of content. Subtracting the mean of the current
    distribution removes that shared component, and cosine similarity starts
    measuring what actually differs. It is two lines of code and, in the
    literature as here, worth several points.
    """
    m = feats.mean(axis=0, keepdims=True) if mean is None else mean
    return l2(feats - m)


def prototype(feats: np.ndarray) -> np.ndarray:
    """One class vector: the mean of the support embeddings, re-normalised."""
    return l2(feats.mean(axis=0, keepdims=True))[0]


def build_bank(encoder, support_crops, patches_per_object: int = 4,
               n_rot: int = 4, centred: bool = True) -> np.ndarray:
    """Support crops -> bank of vectors, augmented by rotation and mirroring.

    Returns the bank rather than a single prototype: keeping N vectors lets the
    caller choose between averaging them ("mean") and a k-NN vote ("knn").
    """
    import cv2

    side = patches_per_object * getattr(encoder, "patch", 14)
    views = []
    for crop in support_crops:
        h, w = crop.shape[:2]
        c = (w / 2.0, h / 2.0)
        for k in range(n_rot):
            m = cv2.getRotationMatrix2D(c, 360.0 * k / n_rot, 1.0)
            rot = cv2.warpAffine(crop, m, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            views.append(rot)
            views.append(cv2.flip(rot, 1))

    feats = []
    for v in views:
        g = encoder.patches(v, (side, side))
        feats.append(g.reshape(-1, g.shape[-1]).mean(axis=0))
    bank = np.stack(feats)
    return centre_l2(bank) if centred else l2(bank)


def score(feats: np.ndarray, bank: np.ndarray, agg: str = "mean",
          topk: int = 3) -> np.ndarray:
    """Cosine similarity of each vector against the bank."""
    sims = feats @ bank.T
    if agg == "mean":
        return sims.mean(axis=1)
    k = min(topk, sims.shape[1])
    return np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)


