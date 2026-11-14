#!/usr/bin/env python3
"""Add --decode peaks: local maxima instead of a global percentile.

    python patch_peaks.py
    python run_dense.py --decode peaks ... --support-dir data/support5 --half b

WHY. The percentile keeps a FIXED SHARE of the grid. At --percentile 98 on a
65x65 grid that is 84 squares, and an object covers about 4x4 = 16 of them, so
a tile can return at most five objects however brightly the rest glow. This
project has measured that ceiling twice: half the misses at --percentile 99
came from it, and most of the gain from hard negatives came from FREEING
budget rather than from removing false alarms.

A local maximum is a local decision. Two objects on a tile do not compete, and
a tile holding eight returns eight. Verified on synthetic maps: with the
percentile the box count stays near its budget whatever is actually there
(10 boxes on pure background, 3 on a tile with 3 objects); with peaks it
tracks the object count.

WHAT STOPS NOISE. Not a threshold on the map, which would be a budget again,
but `min_patches` applied to the GROWN REGION around each peak: a noise spike
dies within a cell or two, a real object holds its value across many.

TWO KNOBS WORTH SWEEPING
    --peak-rel    how far down from the peak the box grows. Lower = bigger
                  boxes. This is what AP@0.5 responds to.
    --peak-floor  percentile below which peaks are ignored. Deliberately loose
                  (90 leaves ~26 objects' worth of room); raise it only if the
                  output is drowning.

Run from the project root. Idempotent; writes nothing if a pattern fails.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

HELPERS = '''def _grow_region(hm, py, px, floor, limit):
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


'''

SIG_OLD = "           neg_weight: float = 1.0) -> tuple[list, list]:"
SIG_NEW = ('           neg_weight: float = 1.0, decode: str = "percentile",\n'
           "           peak_floor: float = 90.0, peak_window: int = 0,\n"
           "           peak_rel: float = 0.6) -> tuple[list, list]:")

BODY_OLD = """    dets = boxes_from_heatmap(cen, raw, tile.shape[:2], percentile,
                              min_patches=min_patches, max_boxes=max_boxes,
                              pad=pad, min_score=min_score)"""
BODY_NEW = """    if decode == "peaks":
        dets = boxes_from_peaks(cen, raw, tile.shape[:2], patches_per_object,
                                floor_pct=peak_floor, window=peak_window,
                                rel=peak_rel, max_boxes=max_boxes, pad=pad,
                                min_score=min_score, min_patches=min_patches)
    else:
        dets = boxes_from_heatmap(cen, raw, tile.shape[:2], percentile,
                                  min_patches=min_patches, max_boxes=max_boxes,
                                  pad=pad, min_score=min_score)"""

MARKER = "def detect(encoder, tile: np.ndarray, bank: np.ndarray, object_px: float,"

ARG_ANCHOR = ('    p.add_argument("--facet", default="token",\n')
ARG_NEW = ('    p.add_argument("--decode", default="percentile",\n'
           '                   choices=["percentile", "peaks"],\n'
           '                   help="how to turn the heatmap into boxes; '
           '\'peaks\' has no per-tile budget")\n'
           '    p.add_argument("--peak-floor", type=float, default=90.0,\n'
           '                   help="--decode peaks: percentile below which '
           'peaks are ignored")\n'
           '    p.add_argument("--peak-window", type=int, default=0,\n'
           '                   help="--decode peaks: neighbourhood for the '
           'local max; 0 derives it from --patches-per-object")\n'
           '    p.add_argument("--peak-rel", type=float, default=0.6,\n'
           '                   help="--decode peaks: how far down from the '
           'peak a box grows; lower = bigger boxes")\n'
           + ARG_ANCHOR)

PASS_OLD = "min_patches=a.min_patches"
PASS_NEW = ("min_patches=a.min_patches, decode=a.decode,\n"
            "                               peak_floor=a.peak_floor,\n"
            "                               peak_window=a.peak_window,\n"
            "                               peak_rel=a.peak_rel")
PASS_OLD2 = "min_patches=args.min_patches"
PASS_NEW2 = ("min_patches=args.min_patches, decode=args.decode,\n"
             "            peak_floor=args.peak_floor,\n"
             "            peak_window=args.peak_window,\n"
             "            peak_rel=args.peak_rel")

SCRIPTS = ["run_dense.py", "benchmark.py", "predict.py", "show_results.py"]


def main() -> int:
    d = ROOT / "src" / "dense.py"
    if not d.exists():
        sys.exit(f"{d} not found - run this from the project root")
    t = d.read_text()

    if "boxes_from_peaks" in t:
        print("  dense.py: already applied")
    else:
        bad = [f"{lbl}: found {t.count(s)}, expected 1"
               for lbl, s in (("signature", SIG_OLD), ("body", BODY_OLD),
                              ("insert point", MARKER)) if t.count(s) != 1]
        if bad:
            sys.exit("ABORTED, nothing written:\n  " + "\n  ".join(bad))
        t = t.replace(SIG_OLD, SIG_NEW, 1)
        t = t.replace(BODY_OLD, BODY_NEW, 1)
        t = t.replace(MARKER, HELPERS + MARKER, 1)
        d.write_text(t)
        print("  dense.py: applied (signature + body + 3 helpers)")

    for name in SCRIPTS:
        p = ROOT / name
        if not p.exists():
            print(f"  {name}: missing, skipped")
            continue
        s = p.read_text()
        if "--decode" in s:
            print(f"  {name}: already applied")
            continue
        if s.count(ARG_ANCHOR) != 1:
            print(f"  {name}: --facet not found, SKIPPED "
                  "(run patch_facet.py first)")
            continue
        s = s.replace(ARG_ANCHOR, ARG_NEW, 1)
        n = s.count(PASS_OLD) + s.count(PASS_OLD2)
        s = s.replace(PASS_OLD, PASS_NEW).replace(PASS_OLD2, PASS_NEW2)
        p.write_text(s)
        print(f"  {name}: applied ({n} call site(s))")

    print("""
done. The sweep - --peak-rel is what AP@0.5 responds to:

  M=facebook/dinov3-vitl16-pretrain-sat493m
  Aa="--support-dir data/support5 --half a --min-patches 3 --quiet"

  printf 'percentile  : '; python run_dense.py --model $M $Aa \\
      --percentile 98 | grep -E 'AP@0.3|tp ' | tr -s ' \\n' ' '; echo
  for r in 0.4 0.6 0.8; do
    printf 'peaks r=%-4s: ' $r
    python run_dense.py --model $M $Aa --decode peaks --peak-rel $r \\
      | grep -E 'AP@0.3|tp ' | tr -s ' \\n' ' '; echo
  done

Reference on half a: AP@0.3 0.720, and the percentile's miss count.
Watch `miss` - that is where a budget shows up.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())