"""Boxes, IGN tile download, tiling, and colour-based pool proposals.

Imagery: IGN Geoplateforme, BD ORTHO(R), Open Licence / Etalab.
Credit "(c) IGN - BD ORTHO(R)" on any figure you publish.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests

from .geo import TILE_PX, deg2tile, resolution

WMTS = "https://data.geopf.fr/wmts"
LAYER_RGB = "ORTHOIMAGERY.ORTHOPHOTOS"
LAYER_IRC = "ORTHOIMAGERY.ORTHOPHOTOS.IRC"
USER_AGENT = "pool-fewshot/1.0 (student project)"


# ------------------------------------------------------------------ boxes

@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int = 0

    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.w() * self.h()

    def clip(self, w: int, h: int) -> "Box":
        return Box(max(0, self.x1), max(0, self.y1),
                   min(w, self.x2), min(h, self.y2), self.cls)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list[Box], scores: list[float], thr: float = 0.35):
    """Plain non-maximum suppression."""
    order = np.argsort(scores)[::-1]
    used = np.zeros(len(boxes), dtype=bool)
    kept_b, kept_s = [], []
    for i in order:
        if used[i]:
            continue
        kept_b.append(boxes[i])
        kept_s.append(scores[i])
        used[i] = True
        for j in order:
            if not used[j] and iou(boxes[i], boxes[j]) > thr:
                used[j] = True
    return kept_b, kept_s


def load_labels(path: Path, w: int, h: int) -> list[Box]:
    """Read YOLO labels: one line per object, `cls cx cy w h` normalised."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
        out.append(Box((cx - bw / 2) * w, (cy - bh / 2) * h,
                       (cx + bw / 2) * w, (cy + bh / 2) * h, cls))
    return out


def save_labels(path: Path, boxes: list[Box], w: int, h: int) -> None:
    lines = [f"{b.cls} {(b.x1 + b.x2) / 2 / w:.6f} {(b.y1 + b.y2) / 2 / h:.6f} "
             f"{b.w() / w:.6f} {b.h() / h:.6f}" for b in boxes]
    Path(path).write_text("\n".join(lines))


# ------------------------------------------------------------------- WMTS

def tile_url(x: int, y: int, z: int, layer: str = LAYER_RGB) -> str:
    return (f"{WMTS}?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
            f"&LAYER={layer}&STYLE=normal&FORMAT=image/jpeg"
            f"&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}")


def fetch_tile(x: int, y: int, z: int, layer: str, cache: Path,
               session: requests.Session, pause: float = 0.1,
               retries: int = 3) -> bytes:
    """Fetch one WMTS tile, with a disk cache so an interrupted run resumes."""
    folder = cache / layer / str(z) / str(x)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{y}.jpg"
    if path.exists():
        return path.read_bytes()

    last = None
    for attempt in range(retries):
        try:
            r = session.get(tile_url(x, y, z, layer),
                            headers={"User-Agent": USER_AGENT}, timeout=30)
            r.raise_for_status()
            path.write_bytes(r.content)
            time.sleep(pause)          # public service: stay polite
            return r.content
        except Exception as exc:       # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"tile {z}/{x}/{y}: {last}")


def fetch_mosaic(bbox: tuple[float, float, float, float], zoom: int,
                 layer: str = LAYER_RGB, cache: str = "data/raw",
                 verbose: bool = True):
    """bbox = (lat_min, lon_min, lat_max, lon_max).

    Returns (BGR image, origin tile) — the origin is needed to turn detection
    pixels back into latitude/longitude.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    x0, y0 = deg2tile(lat_max, lon_min, zoom)
    x1, y1 = deg2tile(lat_min, lon_max, zoom)
    xs = list(range(min(x0, x1), max(x0, x1) + 1))
    ys = list(range(min(y0, y1), max(y0, y1) + 1))
    if verbose:
        print(f"[ign] {len(xs)}x{len(ys)} = {len(xs) * len(ys)} tiles at zoom {zoom}")

    mosaic = np.zeros((len(ys) * TILE_PX, len(xs) * TILE_PX, 3), dtype=np.uint8)
    session = requests.Session()
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            raw = fetch_tile(x, y, zoom, layer, Path(cache), session)
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                mosaic[j * TILE_PX:(j + 1) * TILE_PX,
                       i * TILE_PX:(i + 1) * TILE_PX] = arr
        if verbose:
            print(f"[ign] row {j + 1}/{len(ys)}", end="\r")
    if verbose:
        print(" " * 40, end="\r")
    return mosaic, (min(x0, x1), min(y0, y1))


# ----------------------------------------------------------------- tiling

def cut_tiles(image: np.ndarray, size: int, overlap: int):
    """Cut into overlapping tiles. Yields (name, crop, offset_x, offset_y).

    The name encodes the offset ("t_<ox>_<oy>") so that support/test splits can
    group by SOURCE IMAGE. Splitting by tile would put the same object on both
    sides, because overlapping tiles share their objects.
    """
    h, w = image.shape[:2]
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than size")
    out = []
    for oy in range(0, max(1, h - overlap), step):
        for ox in range(0, max(1, w - overlap), step):
            oy_c, ox_c = min(oy, max(0, h - size)), min(ox, max(0, w - size))
            crop = image[oy_c:oy_c + size, ox_c:ox_c + size]
            if crop.shape[0] < size or crop.shape[1] < size:
                pad = np.zeros((size, size, 3), dtype=image.dtype)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            out.append((f"t_{ox_c}_{oy_c}", crop, ox_c, oy_c))
    return out


def split_half(tile_names, half: str = "all"):
    """Split tiles into two disjoint halves BY SOURCE IMAGE.

    Sources are sorted, then assigned alternately to "a" and "b". Every tile
    from one source lands in the same half.

    Why this matters. Support crops are chosen by ranking candidates on real
    tiles. If that ranking runs on the same tiles the model is later evaluated
    on, the crops were picked *because* they score well there, and the reported
    number is inflated — a selection leak, not a result. Rank on half "a",
    evaluate on half "b", and the number means something.

    Splitting by tile instead of by source would not be enough: tiles overlap,
    so the same pool can appear on both sides.
    """
    if half not in ("a", "b", "all"):
        raise ValueError("half must be 'a', 'b' or 'all'")
    if half == "all":
        return list(tile_names)
    sources = sorted({source_of(Path(str(n)).stem) for n in tile_names})
    want = {s for i, s in enumerate(sources) if (i % 2 == 0) == (half == "a")}
    return [n for n in tile_names if source_of(Path(str(n)).stem) in want]


def source_of(tile_name: str) -> str:
    """Tile name -> source image, used by the support/test split.

    "t_512_256" -> "t"          (a cut tile: drop the _ox_oy suffix)
    "pool_0042" -> "pool_0042"  (a standalone image is its own source)

    The suffix is only dropped when BOTH trailing fields are numeric. Otherwise
    an external dataset named "image_0042.jpg" would collapse into one source
    and the split would silently break.
    """
    parts = tile_name.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return tile_name


# ------------------------------------------------- colour pre-annotation

def propose_pools(tile: np.ndarray, res_m: float, min_m2: float = 8.0,
                  max_m2: float = 200.0, max_per_tile: int = 12) -> list[Box]:
    """Propose swimming pools by colour. Three filters, first one matters most.

    1. BRIGHTNESS. Filtering on hue alone fails: vegetation shadows fall in the
       same cyan-blue range as water. Measured on real imagery, the V >= 110
       threshold alone cuts the mask from 10.7% of pixels to 0.19% — 56x fewer.
       Pool water is BRIGHT, a shadow is dark.
    2. BLUE DOMINANCE. b - r >= 45: water is strongly blue, dark vegetation
       only weakly so (measured median 18).
    3. COMPACTNESS. A pool fills its bounding box; a shadow blob broken up by
       branches does not. extent >= 0.55.

    WARNING: these proposals are NOT ground truth. They find clean blue pools.
    You must manually ADD the covered, green, shaded and empty ones, otherwise
    you end up measuring a colour threshold rather than few-shot matching.
    """
    b, g, r = (tile[:, :, i].astype(np.int16) for i in range(3))
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = (hsv[:, :, i].astype(np.int16) for i in range(3))

    mask = ((h_ >= 85) & (h_ <= 112) & (s_ >= 70) & (v_ >= 110)
            & ((b - r) >= 45)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    px_min, px_max = min_m2 / res_m ** 2, max_m2 / res_m ** 2
    found = []
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (px_min <= area <= px_max):
            continue
        if max(w, h) / max(1, min(w, h)) > 3.0:
            continue
        if area / max(1, w * h) < 0.55:
            continue
        found.append((area, Box(float(x), float(y),
                                float(x + w), float(y + h), 0)))
    found.sort(key=lambda t: -t[0])
    return [b for _a, b in found[:max_per_tile]]


def suggest_scales(zoom: int, lat: float) -> tuple[list[int], list[float]]:
    """Window scales and aspect ratios sized for an 8x4 m pool."""
    r = resolution(zoom, lat)
    typical = (8 * 4) ** 0.5 / r
    scales = sorted({max(8, int(round(typical * f))) for f in (0.6, 1.0, 1.6)})
    return scales, [1.0, 2.0]
