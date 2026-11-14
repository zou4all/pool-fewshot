"""Drawing: annotated tiles and result sheets."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

GREEN = (0, 200, 0)      # correct detection
RED = (0, 0, 255)        # false alarm
BLUE = (255, 160, 0)     # missed object (ground truth with no match)
WHITE = (255, 255, 255)


def _xyxy(b) -> tuple[int, int, int, int]:
    """Accept a Box/Det dataclass or a plain (x1, y1, x2, y2) tuple."""
    if hasattr(b, "x1"):
        vals = (b.x1, b.y1, b.x2, b.y2)
    else:
        vals = (b[0], b[1], b[2], b[3])
    return tuple(int(round(float(v))) for v in vals)  # type: ignore[return-value]


def draw_boxes(image: np.ndarray, boxes, scores=None, colour=GREEN,
               thickness: int = 2, labels=None, dashed: bool = False):
    """Draw boxes on a COPY of the image."""
    out = image.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = _xyxy(b)
        if dashed:
            _dashed_rect(out, (x1, y1), (x2, y2), colour, thickness)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        text = None
        if labels is not None and i < len(labels):
            text = str(labels[i])
        elif scores is not None and i < len(scores):
            text = f"{scores[i]:.2f}"
        if text:
            _label(out, text, x1, y1, colour)
    return out


def _dashed_rect(img, p1, p2, colour, thickness=2, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), colour, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), colour, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), colour, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), colour, thickness)


def _label(img, text, x, y, colour):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), colour, -1)
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA)


def title_bar(image: np.ndarray, text: str, height: int = 26) -> np.ndarray:
    """Add a black caption bar above the image."""
    bar = np.zeros((height, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text[:80], (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def contact_sheet(images, cols: int = 4, cell: int = 320,
                  captions=None) -> np.ndarray:
    """Grid of thumbnails — the fastest way to see what a model is doing."""
    if not images:
        return np.zeros((cell, cell, 3), dtype=np.uint8)
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * (cell + 26), cols * cell, 3), 20, dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        thumb = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        cap = captions[i] if captions and i < len(captions) else ""
        thumb = title_bar(thumb, cap)
        sheet[r * (cell + 26):(r + 1) * (cell + 26),
              c * cell:(c + 1) * cell] = thumb
    return sheet


def heatmap_overlay(tile: np.ndarray, hm: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Superimpose the similarity map on the tile (blue = low, red = high)."""
    span = float(np.ptp(hm))
    norm = (hm - hm.min()) / (span + 1e-8)
    big = cv2.resize((norm * 255).astype(np.uint8),
                     (tile.shape[1], tile.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
    colour = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    return cv2.addWeighted(tile, 1 - alpha, colour, alpha, 0)


def save(path, image) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
