"""Few-shot detection through a dense similarity heatmap.

The few-shot part is standard (prototype + cosine similarity). What differs
from a sliding window: the tile goes through the network ONCE and the model
returns one vector per `patch`-pixel square. Every square is compared to the
prototype -> heatmap -> threshold -> connected components -> boxes.

VARIABLE PATCH SIZE
DINOv2 uses 14-pixel patches, DINOv3 uses 16. The value is read from
`model.config.patch_size` and carried on the encoder (`encoder.patch`), never
hard-coded: a wrong value would shift the grid with no error raised, and the
results would be silently wrong.

REGISTER TOKENS
DINOv3 adds 4 register tokens after the CLS token, SigLIP has no CLS token at
all. The count is derived from the difference between the number of tokens and
the number of expected squares, so the code works in all three cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

PATCH_DEFAULT = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def patch_of(encoder) -> int:
    return int(getattr(encoder, "patch", PATCH_DEFAULT))


def round_patch(v: int, patch: int) -> int:
    return max(2 * patch, int(round(v / patch)) * patch)


class DenseEncoder:
    """Vision backbone returning the patch grid.

    Known-good model ids:
        facebook/dinov2-base                       (patch 14, 86M)
        facebook/dinov2-with-registers-base        (patch 14, 86M)
        facebook/dinov3-vitb16-pretrain-lvd1689m   (patch 16, 86M, web)
        facebook/dinov3-vitl16-pretrain-sat493m    (patch 16, 300M, satellite)
        google/siglip2-base-patch16-224            (patch 16, no CLS token)

    DINOv3 weights are gated on HuggingFace: accept the licence on the model
    page and run `hf auth login` first.

    RTX 50xx (Blackwell) needs a cu128 torch build:
        pip install torch --index-url https://download.pytorch.org/whl/cu128
    """

    def __init__(self, model: str = "facebook/dinov2-base",
                 device: str | None = None, fp16: bool = True,
                 trust_remote_code: bool = False):
        import inspect

        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model_name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device.startswith("cuda")
        self.model = AutoModel.from_pretrained(
            model, trust_remote_code=trust_remote_code).to(self.device).eval()
        if self.fp16:
            self.model = self.model.half()

        # SigLIP and CLIP expose an image+text model: keep the vision tower
        # only, otherwise the forward pass also expects text tokens.
        self.net = getattr(self.model, "vision_model", self.model)
        cfg = getattr(self.net, "config", self.model.config)
        vcfg = getattr(self.model.config, "vision_config", None) or cfg
        self.patch = int(getattr(vcfg, "patch_size", PATCH_DEFAULT))
        self.dim = int(getattr(vcfg, "hidden_size", 0))

        # DINOv2/v3 interpolate position encodings on their own. SigLIP and
        # CLIP are frozen at their training resolution and must be asked
        # explicitly, otherwise any size other than 224 fails.
        try:
            self._interp = "interpolate_pos_encoding" in inspect.signature(
                self.net.forward).parameters
        except (TypeError, ValueError):
            self._interp = False

    @property
    def name(self) -> str:
        return self.model_name.split("/")[-1]

    def patches(self, img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        """Return the (gh, gw, dim) patch grid, L2-normalised."""
        torch = self.torch
        h, w = size_hw
        if h % self.patch or w % self.patch:
            raise ValueError(f"size {w}x{h} is not a multiple of {self.patch}")
        interp = cv2.INTER_AREA if img.shape[0] > h else cv2.INTER_LINEAR
        rgb = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=interp),
                           cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0)
            t = t.float().div_(255.0)
            t = (t - self._mean) / self._std
            if self.fp16:
                t = t.half()
            kw = {"pixel_values": t}
            if self._interp:
                kw["interpolate_pos_encoding"] = True
            tok = self.net(**kw).last_hidden_state[0]

        if not torch.isfinite(tok).all():
            # Half-precision overflow (dinov2-large notably). Without this
            # guard NaNs propagate silently: NaN >= threshold is False
            # everywhere, the mask is empty, no box is produced, and AP comes
            # out as exactly 0.000 with no error message at all.
            if self.fp16:
                print(f"[{self.name}] NaN in fp16 -> falling back to fp32 "
                      "(slower, more memory)")
                self.fp16 = False
                self.model = self.model.float()
                self.net = getattr(self.model, "vision_model", self.model)
                return self.patches(img, size_hw)
            raise RuntimeError(f"{self.name} produces NaNs even in fp32")

        # 0 extra tokens for SigLIP, 1 for DINOv2 (CLS), 5 for DINOv3
        # (CLS + 4 registers). Extra tokens always come FIRST.
        n_exp = (h // self.patch) * (w // self.patch)
        extra = tok.shape[0] - n_exp
        if extra < 0:
            raise RuntimeError(f"{tok.shape[0]} tokens for {n_exp} expected "
                               f"squares: inconsistent patch size "
                               f"(patch={self.patch})")
        grid = tok[extra:].float().cpu().numpy()
        return l2(grid.reshape(h // self.patch, w // self.patch, -1))

    @property
    def _mean(self):
        if not hasattr(self, "_mean_t"):
            self._mean_t = self.torch.tensor(
                IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        return self._mean_t

    @property
    def _std(self):
        if not hasattr(self, "_std_t"):
            self._std_t = self.torch.tensor(
                IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return self._std_t

    def encode_crops(self, crops: Sequence[np.ndarray],
                     patches_per_object: int = 4) -> np.ndarray:
        """One mean-pooled vector per crop. Used by the sliding-window baseline."""
        side = patches_per_object * self.patch
        out = []
        for c in crops:
            g = self.patches(c, (side, side))
            out.append(g.reshape(-1, g.shape[-1]).mean(axis=0))
        return l2(np.stack(out))


# Backwards-compatible alias: earlier versions of this project called it
# DenseDino, before non-DINO backbones were supported.
DenseDino = DenseEncoder


class RawPixelEncoder:
    """Control witness: 16x16 downscaled RGB instead of a neural backbone.

    Same interface as DenseEncoder, no learned weights, no download. If the
    real backbone does not clearly beat this, the pipeline is measuring colour,
    not features. Worth keeping: it is the cheapest way to catch a plumbing bug
    that quietly turns a "few-shot detector" into a blue-pixel threshold.

    Every script accepts it as `--model raw-pixels`, which also makes the whole
    chain runnable on a machine with no GPU and no network.
    """

    patch = 14
    dim = 48
    model_name = "raw-pixels"
    device = "cpu"
    fp16 = False

    @property
    def name(self) -> str:
        return "raw-pixels"

    def patches(self, img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        h, w = size_hw
        gh, gw = h // self.patch, w // self.patch
        small = cv2.resize(img, (gw * 4, gh * 4), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32) / 255.0
        grid = small.reshape(gh, 4, gw, 4, 3).transpose(0, 2, 1, 3, 4)
        return l2(grid.reshape(gh, gw, -1))

    def encode_crops(self, crops: Sequence[np.ndarray],
                     patches_per_object: int = 4) -> np.ndarray:
        side = patches_per_object * self.patch
        out = []
        for c in crops:
            g = self.patches(c, (side, side))
            out.append(g.reshape(-1, g.shape[-1]).mean(axis=0))
        return l2(np.stack(out))


def make_encoder(model: str = "facebook/dinov2-base", **kw):
    """Encoder factory. `raw-pixels` returns the control witness."""
    if model in ("raw-pixels", "raw", "control"):
        return RawPixelEncoder()
    return DenseEncoder(model, **kw)


# -------------------------------------------------------------- prototype

def dense_prototype(encoder, support: Sequence[np.ndarray],
                    patches_per_object: int = 4, n_rot: int = 4) -> np.ndarray:
    """Bank of embeddings describing the target object.

    Each support crop is resized so it spans patches_per_object squares,
    rotated, encoded, then mean-pooled. Seen from above an object has no
    canonical orientation: without rotations the prototype would encode the
    orientation of the support crops as much as the object itself.
    """
    side = patches_per_object * patch_of(encoder)
    bank = []
    for crop in support:
        h, w = crop.shape[:2]
        c = (w / 2.0, h / 2.0)
        for k in range(n_rot):
            m = cv2.getRotationMatrix2D(c, 360.0 * k / n_rot, 1.0)
            rot = cv2.warpAffine(crop, m, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            for v in (rot, cv2.flip(rot, 1)):
                g = encoder.patches(v, (side, side))
                bank.append(g.reshape(-1, g.shape[-1]).mean(axis=0))
    return l2(np.stack(bank))


def target_size(tile_hw: tuple[int, int], object_px: float,
                patches_per_object: int, patch: int,
                max_side: int = 1040) -> tuple[int, int]:
    """Input size so that an object spans patches_per_object squares.

    This is the single most important setting: a square must see the SAME
    amount of object on the support side and on the target side. Otherwise a
    close-up pool is compared to a distant pool and similarity collapses.
    """
    h, w = tile_hw
    scale = (patches_per_object * patch) / max(4.0, object_px)
    th = round_patch(int(h * scale), patch)
    tw = round_patch(int(w * scale), patch)
    if max(th, tw) > max_side:
        f = max_side / max(th, tw)
        th, tw = round_patch(int(th * f), patch), round_patch(int(tw * f), patch)
    return th, tw


def _aggregate(sims: np.ndarray, agg: str, topk: int) -> np.ndarray:
    """Reduce per-square similarities against the bank into one score.

    "mean": average over the WHOLE bank. A bad support vector weighs 1/N and
        cannot dominate. Preferred for a visually homogeneous class (pools).
    "knn": average of the topk best. Preserves the modes of a heterogeneous
        class, but a single generic vector that resembles everything saturates
        the topk everywhere and flattens the map.
    """
    if agg == "mean":
        return sims.mean(axis=1)
    k = min(topk, sims.shape[1])
    return np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)


def heatmap(encoder, tile: np.ndarray, bank: np.ndarray, object_px: float,
            patches_per_object: int = 4, topk: int = 3,
            agg: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Return (centred map, raw map).

    The CENTRED map PLACES the boxes: subtracting the tile's own background
    makes objects stand out. The RAW map SCORES them: without centring the
    scores stay comparable across tiles, which matters when every detection is
    ranked together (global AP, FROC).
    """
    p = patch_of(encoder)
    th, tw = target_size(tile.shape[:2], object_px, patches_per_object, p)
    grid = encoder.patches(tile, (th, tw))
    gh, gw, d = grid.shape
    flat = grid.reshape(-1, d)

    raw = _aggregate(flat @ bank.T, agg, topk)
    mu = flat.mean(axis=0, keepdims=True)
    cen = _aggregate(l2(flat - mu) @ l2(bank - mu).T, agg, topk)
    return cen.reshape(gh, gw), raw.reshape(gh, gw)


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


def boxes_from_heatmap(hm: np.ndarray, raw: np.ndarray,
                       tile_hw: tuple[int, int], percentile: float = 99.0,
                       min_patches: int = 1, max_boxes: int = 30,
                       pad: float = 0.08, min_score: float = -1.0) -> list[Det]:
    """Threshold the centred map, score with the raw map.

    The threshold is a PERCENTILE, so it adapts per tile — which means an empty
    tile still produces boxes. That is intended: those are the false alarms
    FROC is meant to measure, and their raw score stays low so they sink in the
    global ranking. Use min_score to drop them entirely.

    NOTE: percentile 99 on a 53x53 grid keeps ~28 squares. A pool spans about
    4x4 = 16 squares, so the budget only fits one or two pools per tile. On
    tiles with more objects, lower the percentile (97 gives ~84 squares).
    """
    h, w = tile_hw
    gh, gw = hm.shape
    mask = (hm >= float(np.percentile(hm, percentile))).astype(np.uint8)
    if mask.sum() == 0:
        return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sy, sx = h / gh, w / gw
    out: list[Det] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_patches:
            continue
        score = float(raw[lab == i].max())
        if score < min_score:
            continue
        x1, y1, x2, y2 = x * sx, y * sy, (x + bw) * sx, (y + bh) * sy
        mx, my = (x2 - x1) * pad, (y2 - y1) * pad
        out.append(Det(max(0, x1 - mx), max(0, y1 - my),
                       min(w, x2 + mx), min(h, y2 + my), score))
    out.sort(key=lambda d: -d.score)
    return out[:max_boxes]


def detect(encoder, tile: np.ndarray, bank: np.ndarray, object_px: float,
           patches_per_object: int = 4, percentile: float = 99.0,
           max_boxes: int = 30, pad: float = 0.08, min_score: float = -1.0,
           agg: str = "mean", topk: int = 3,
           min_patches: int = 1) -> tuple[list, list]:
    """One network pass, one heatmap, a list of boxes.

    min_patches is the speckle filter: the smallest connected component that
    may become a box. At 1 a single isolated square is a detection, which is
    how a weak backbone turns its percentile budget into dozens of specks. A
    real object covers patches_per_object^2 squares or more, so requiring 3-4
    costs almost no recall and removes most of the noise.
    """
    cen, raw = heatmap(encoder, tile, bank, object_px, patches_per_object,
                       topk=topk, agg=agg)
    dets = boxes_from_heatmap(cen, raw, tile.shape[:2], percentile,
                              min_patches=min_patches, max_boxes=max_boxes,
                              pad=pad, min_score=min_score)
    return [(d.x1, d.y1, d.x2, d.y2) for d in dets], [d.score for d in dets]
