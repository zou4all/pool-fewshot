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


# ------------------------------------------------- facets (keys / queries)

FACET_ORDER = {"query": 0, "key": 1, "value": 2}
FACET_NAMES = {
    "key": ("key", "k_proj", "to_k"),
    "query": ("query", "q_proj", "to_q"),
    "value": ("value", "v_proj", "to_v"),
}
FUSED_NAMES = ("qkv", "in_proj", "qkv_proj")
ATTN_LEAVES = ("key", "query", "value", "k_proj", "q_proj", "v_proj",
               "to_k", "to_q", "to_v", "qkv", "in_proj", "qkv_proj")


def _has_attention(block, nn) -> bool:
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in ATTN_LEAVES:
            return True
    return False


def find_blocks(net):
    """The transformer block stack, found by STRUCTURE rather than by name.

    Hard-coded paths break the moment an architecture nests differently.
    DINOv3 puts its blocks under `.model.layers`, which an earlier version of
    this function did not look for: it returned [], the encoder fell back to
    tokens, and the whole facet comparison measured nothing.

    So: walk every ModuleList and keep the longest one whose entries ALL
    contain an attention projection. That is the block stack in any ViT,
    whatever the nesting, and a decoy ModuleList of plain Linears cannot be
    mistaken for it.
    """
    import torch
    nn = torch.nn
    best = []
    for _name, mod in net.named_modules():
        if not isinstance(mod, (nn.ModuleList, nn.Sequential)) or len(mod) < 2:
            continue
        if len(mod) > len(best) and all(_has_attention(b, nn) for b in mod):
            best = list(mod)
    return best


def find_projection(block, facet, torch):
    """Locate the Linear producing `facet` inside one block.

    Returns (module, fused_facet_or_None, name). A dedicated projection wins
    over a fused qkv: slicing a fused tensor is one more assumption, so it is
    used only when there is no alternative.

        DINOv2   attention.attention.key   dedicated
        DINOv3   attention.qkv             fused, keys are the middle third
        SigLIP   self_attn.k_proj          dedicated
    """
    wanted = FACET_NAMES[facet]
    fused = None
    for name, mod in block.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in wanted:
            return mod, None, name
        if leaf in FUSED_NAMES and fused is None:
            fused = (mod, facet, name)
    return fused if fused else (None, None, "")


def slice_fused(out, facet):
    """Fused qkv output -> only the facet asked for.

    Two layouts exist: (B, N, 3*D) with q|k|v contiguous, and (B, N, 3, H, D).
    Anything else raises rather than returning a silently wrong slice.
    """
    if out.dim() == 3:
        if out.shape[-1] % 3:
            raise RuntimeError(f"fused projection width {out.shape[-1]} is not "
                               "divisible by 3 - cannot locate the keys")
        d = out.shape[-1] // 3
        i = FACET_ORDER[facet]
        return out[..., i * d:(i + 1) * d]
    if out.dim() == 5 and out.shape[2] == 3:
        return out[:, :, FACET_ORDER[facet]].flatten(2)
    raise RuntimeError(f"unexpected fused qkv shape {tuple(out.shape)}")


class _FacetHook:
    """Capture one block's facet during a forward pass.

    A context manager so the hook is always removed even if the forward
    raises: a leaked hook would quietly corrupt every later call.
    """

    def __init__(self, module, fused_facet):
        self.module, self.fused, self.out, self._h = module, fused_facet, None, None

    def __enter__(self):
        def grab(_m, _i, o):
            self.out = o[0] if isinstance(o, tuple) else o
        self._h = self.module.register_forward_hook(grab)
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.remove()
            self._h = None
        return False

    def tokens(self):
        if self.out is None:
            raise RuntimeError("the facet hook captured nothing - the module "
                               "was found but never called")
        out = slice_fused(self.out, self.fused) if self.fused else self.out
        return out[0] if out.dim() == 3 else out


def block_index(layer, n: int) -> int:
    """--layer value -> 0-based index into the block list.

    Kept consistent with the hidden_states convention: 1 is the first block,
    -1 the last, a float is a fraction of the depth, None means the last.
    """
    if layer is None:
        return n - 1
    if isinstance(layer, float):
        return max(0, min(n - 1, int(round(layer * n)) - 1))
    i = int(layer)
    if i < 0:
        return max(0, n + i)
    return max(0, min(n - 1, i - 1))


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
                 trust_remote_code: bool = False, load_in: str | None = None,
                 layer: int | float | None = None,
                 facet: str = "token"):
        """load_in: None | "8bit" | "4bit".

        Quantisation is what puts a model bigger than your GPU inside it.
        Weights in fp16 cost 2 bytes per parameter, so a 6.7B ViT needs 13.4 GB
        before a single activation exists. 8-bit halves that, NF4 quarters it:
        the same model lands at ~3.4 GB and fits in 8 GB with room to spare.

        It is not free. Quantising shifts the features, and nobody has measured
        what that costs a cosine-similarity pipeline like this one. Measure it
        before you trust it: run a model that DOES fit in both fp16 and 4-bit
        (`dinov3-vitl16-pretrain-sat493m` is the obvious choice) and compare.
        That difference is your error bar on any quantised result.

        Needs `pip install bitsandbytes accelerate`.

        layer: which transformer block to read features from.
            None (default) = the final output, after the model's last norm.
            An int indexes hidden_states: 1 is the first block, -1 the last.
            A float is a fraction of the depth: 0.75 = three quarters in.

            NOTE: `--layer -1` is NOT identical to the default. The default
            uses last_hidden_state, which has the final LayerNorm applied;
            hidden_states[-1] is the same block before that norm. If the two
            differ a lot, the final norm is doing real work and that is worth
            knowing on its own.
        """
        import inspect

        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model_name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quant = load_in
        self.layer = layer
        self.n_layers = None
        self.facet = facet
        self._facet_mod = None
        self._facet_fused = None

        if load_in in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            if not self.device.startswith("cuda"):
                raise RuntimeError("quantisation needs a CUDA device")
            if load_in == "4bit":
                qc = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True)
            else:
                qc = BitsAndBytesConfig(load_in_8bit=True)
            # No .to() and no .half() here: bitsandbytes places and casts the
            # weights itself, and calling either on a quantised model fails.
            self.model = AutoModel.from_pretrained(
                model, quantization_config=qc, torch_dtype=torch.float16,
                device_map={"": self.device},
                trust_remote_code=trust_remote_code).eval()
            self.fp16 = True          # inputs still go in as fp16
        else:
            self.model = AutoModel.from_pretrained(
                model, trust_remote_code=trust_remote_code).to(self.device).eval()
            self.fp16 = fp16 and self.device.startswith("cuda")
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

        # Locate the facet projection once, at construction. If the
        # architecture is not recognised, say so loudly and fall back to
        # tokens - a silent fallback would make the comparison meaningless.
        if facet != "token":
            blocks = find_blocks(self.net)
            if not blocks:
                print(f"[facet] {self.name}: block list not found, "
                      "falling back to tokens")
                self.facet = "token"
            else:
                i = block_index(layer, len(blocks))
                mod, fused, path = find_projection(blocks[i], facet, torch)
                if mod is None:
                    print(f"[facet] {self.name}: no '{facet}' projection in "
                          f"block {i + 1}/{len(blocks)}, falling back to tokens")
                    self.facet = "token"
                else:
                    self._facet_mod, self._facet_fused = mod, fused
                    self.n_layers = len(blocks)
                    kind = "fused qkv" if fused else "dedicated"
                    print(f"[facet] {facet} <- block {i + 1}/{len(blocks)}, "
                          f"{path} ({kind})")

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
            if self._facet_mod is not None:
                # The facet hook reads inside the chosen block, so the block
                # index already selected it - `layer` must not also index
                # hidden_states here.
                with _FacetHook(self._facet_mod, self._facet_fused) as _h:
                    self.net(**kw)
                tok = _h.tokens()
            elif self.layer is None:
                tok = self.net(**kw).last_hidden_state[0]
            else:
                # Read an intermediate block instead of the final one. The last
                # layer is the most abstract: good at saying "this is a pool",
                # worse at saying exactly where. Deeper models push that
                # abstraction further, which is the likeliest reason the giants
                # underperform here.
                kw["output_hidden_states"] = True
                hs = self.net(**kw).hidden_states
                self.n_layers = len(hs) - 1        # hs[0] is the embedding
                idx = self.layer
                if isinstance(idx, float):          # 0.75 -> three quarters in
                    idx = max(1, min(self.n_layers,
                                     int(round(idx * self.n_layers))))
                tok = hs[idx][0]

        if not torch.isfinite(tok).all():
            # Half-precision overflow (dinov2-large notably). Without this
            # guard NaNs propagate silently: NaN >= threshold is False
            # everywhere, the mask is empty, no box is produced, and AP comes
            # out as exactly 0.000 with no error message at all.
            if self.quant:
                raise RuntimeError(
                    f"{self.name} produces NaNs in {self.quant}. A quantised "
                    "model cannot be converted back, so there is no fallback: "
                    "re-run without --load-4bit/--load-8bit, or try 8bit if "
                    "you were on 4bit.")
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
    quant = None
    layer = None
    n_layers = None
    facet = "token"
    _facet_mod = None
    _facet_fused = None

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


def vram_estimate(params_b: float, bits: int = 16) -> float:
    """Rough GB of VRAM the weights alone need. Activations come on top."""
    return params_b * 1e9 * bits / 8 / 1024 ** 3


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
            patches_per_object: int = 4, topk: int = 3, agg: str = "mean",
            max_side: int = 1040, neg_bank: np.ndarray | None = None,
            neg_weight: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (centred map, raw map).

    The CENTRED map PLACES the boxes: subtracting the tile's own background
    makes objects stand out. The RAW map SCORES them: without centring the
    scores stay comparable across tiles, which matters when every detection is
    ranked together (global AP, FROC).
    """
    p = patch_of(encoder)
    th, tw = target_size(tile.shape[:2], object_px, patches_per_object, p,
                         max_side=max_side)
    grid = encoder.patches(tile, (th, tw))
    gh, gw, d = grid.shape
    flat = grid.reshape(-1, d)

    raw = _aggregate(flat @ bank.T, agg, topk)
    mu = flat.mean(axis=0, keepdims=True)
    cen = _aggregate(l2(flat - mu) @ l2(bank - mu).T, agg, topk)

    if neg_bank is not None and len(neg_bank):
        # Subtract how much each square resembles a known FALSE alarm.
        #
        # The positive prototype answers "does this look like a pool?". On this
        # data that question is nearly solved - recall is 86% - and what costs
        # precision is that bright rectangular roofs and terraces answer "yes"
        # too. A negative bank, mined from tiles that contain no pool at all,
        # answers the second question: "does this also look like the things
        # that keep fooling me?".
        #
        # MAX, not mean: a square is suspicious if it resembles ANY hard
        # negative. Averaging over the bank would let one odd negative dilute
        # the very confusion the bank exists to catch.
        raw = raw - neg_weight * (flat @ neg_bank.T).max(axis=1)
        cen = cen - neg_weight * (l2(flat - mu) @ l2(neg_bank - mu).T).max(axis=1)

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


def _grow_region(hm, py, px, floor, limit):
    """Flood from a peak over cells above `floor`, bounded by `limit` cells.

    Returns (y0, y1, x0, x1, n_cells). The cell count is what separates a real
    object from a noise spike.
    """
    from collections import deque
    gh, gw = hm.shape
    seen = {(py, px)}
    q = deque([(py, px)])
    y0 = y1 = py
    x0 = x1 = px
    while q and len(seen) <= limit:
        cy, cx = q.popleft()
        y0, y1 = min(y0, cy), max(y1, cy)
        x0, x1 = min(x0, cx), max(x1, cx)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < gh and 0 <= nx < gw and (ny, nx) not in seen
                    and hm[ny, nx] >= floor):
                seen.add((ny, nx))
                q.append((ny, nx))
    return y0, y1, x0, x1, len(seen)


def boxes_from_peaks(hm, raw, tile_hw, patches_per_object: int = 4,
                     floor_pct: float = 90.0, window: int = 0,
                     rel: float = 0.6, max_boxes: int = 30, pad: float = 0.08,
                     min_score: float = -1.0, nms_thr: float = 0.35,
                     min_patches: int = 3):
    """Every local maximum is one candidate object - no global budget.

    See patch_peaks.py for why this exists. The short version: a percentile
    keeps a fixed share of the grid, so objects compete for room on a crowded
    tile; a local maximum does not.
    """
    gh, gw = hm.shape
    th, tw = tile_hw
    sy, sx = th / gh, tw / gw

    if window <= 0:                 # one peak per object at most
        window = max(3, int(patches_per_object) | 1)
    dil = cv2.dilate(hm.astype(np.float32),
                     np.ones((window, window), np.uint8))
    floor = float(np.percentile(hm, floor_pct))
    ys, xs = np.where((hm >= dil - 1e-9) & (hm >= floor))
    if len(ys) == 0:
        return []

    order = np.argsort([-raw[y, x] for y, x in zip(ys, xs)])
    limit = max(4, (patches_per_object * 3) ** 2)
    out = []
    for i in order:
        py, px = int(ys[i]), int(xs[i])
        s = float(raw[py, px])
        if s < min_score:
            continue
        # Grow down to a fraction of the peak's CONTRAST above the floor, not
        # of its absolute value. Cosine similarities sit in a narrow band well
        # above zero - background ~0.37, a pool ~0.45 - so a multiplicative
        # threshold like 0.6 * peak lands under the background and the flood
        # swallows the whole tile.
        y0, y1, x0, x1, n_cells = _grow_region(
            hm, py, px, floor + rel * (hm[py, px] - floor), limit)
        if n_cells < min_patches:
            continue                # a spike, not an object
        bx1, by1 = x0 * sx, y0 * sy
        bx2, by2 = (x1 + 1) * sx, (y1 + 1) * sy
        mx, my = (bx2 - bx1) * pad, (by2 - by1) * pad
        d = Det(max(0.0, bx1 - mx), max(0.0, by1 - my),
                min(float(tw), bx2 + mx), min(float(th), by2 + my), s)
        if all(_box_iou(d, k) <= nms_thr for k in out):
            out.append(d)
        if len(out) >= max_boxes:
            break
    return out


def _box_iou(a, b) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = ((a.x2 - a.x1) * (a.y2 - a.y1)
          + (b.x2 - b.x1) * (b.y2 - b.y1) - inter)
    return inter / ua if ua > 0 else 0.0


def detect(encoder, tile: np.ndarray, bank: np.ndarray, object_px: float,
           patches_per_object: int = 4, percentile: float = 99.0,
           max_boxes: int = 30, pad: float = 0.08, min_score: float = -1.0,
           agg: str = "mean", topk: int = 3,
           min_patches: int = 1, max_side: int = 1040,
           neg_bank: np.ndarray | None = None,
           neg_weight: float = 1.0, decode: str = "percentile",
           peak_floor: float = 90.0, peak_window: int = 0,
           peak_rel: float = 0.6) -> tuple[list, list]:
    """One network pass, one heatmap, a list of boxes.

    max_side caps the resized tile, and therefore the VRAM the forward pass
    needs: dropping 1040 -> 720 more than halves the number of patches. It
    also changes the apparent object size, so a run at a different max_side is
    NOT comparable to the others. Lower it for every model in a comparison or
    for none.

    min_patches is the speckle filter: the smallest connected component that
    may become a box. At 1 a single isolated square is a detection, which is
    how a weak backbone turns its percentile budget into dozens of specks. A
    real object covers patches_per_object^2 squares or more, so requiring 3-4
    costs almost no recall and removes most of the noise.
    """
    cen, raw = heatmap(encoder, tile, bank, object_px, patches_per_object,
                       topk=topk, agg=agg, max_side=max_side,
                       neg_bank=neg_bank, neg_weight=neg_weight)
    if decode == "peaks":
        dets = boxes_from_peaks(cen, raw, tile.shape[:2], patches_per_object,
                                floor_pct=peak_floor, window=peak_window,
                                rel=peak_rel, max_boxes=max_boxes, pad=pad,
                                min_score=min_score, min_patches=min_patches)
    else:
        dets = boxes_from_heatmap(cen, raw, tile.shape[:2], percentile,
                                  min_patches=min_patches, max_boxes=max_boxes,
                                  pad=pad, min_score=min_score)
    return [(d.x1, d.y1, d.x2, d.y2) for d in dets], [d.score for d in dets]
