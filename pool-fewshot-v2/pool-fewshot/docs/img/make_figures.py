#!/usr/bin/env python3
"""Regenerate the README figures from the measured numbers.

    python docs/img/make_figures.py

Every value below is copied from a run reported in docs/RESULTS.md. Nothing is
computed from the model here, so the figures cannot drift from the tables
unless someone edits one and not the other.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = pathlib.Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"      # the thing being shown / decision side
ORANGE = "#eb6834"    # feature side
MUTED = "#b9b8b2"     # context
NOISE = 0.07          # measured AP@0.3 noise floor, docs/RESULTS.md section 2

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})


def clean(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------ 1. the arc
def arc():
    steps = ["original\ndefaults", "percentile 98\nmin-patches 3",
             "+ hard\nnegatives", "+ local\nmaxima"]
    ap = [0.659, 0.783, 0.875, 0.890]
    miss = [35, 20, 9, 6]
    colors = [MUTED, MUTED, MUTED, BLUE]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, vals, title, fmt, top in (
            (a, ap, "AP@0.3  (higher is better)", "{:.3f}", 1.0),
            (b, miss, "Missed objects, out of 146  (lower is better)", "{}", 40)):
        bars = ax.bar(steps, vals, width=0.58, color=colors)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + top * 0.02, fmt.format(v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(0, top)
        ax.set_title(title, loc="left", fontsize=11, color=INK2, pad=10)
        clean(ax)
    fig.suptitle("Same backbone, same five crops — only the decision changes",
                 x=0.02, ha="left", fontsize=13.5, fontweight="bold", y=1.04)
    fig.text(0.02, -0.04,
             "Half b, 102 tiles, 146 objects, dinov3-vitl16-sat493m. No training, "
             "no new annotation between any two bars.",
             fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, "arc.png")


# ------------------------------------------------------------- 2. backbones
def backbones():
    rows = [  # (label, params, AP@0.3, pretraining)
        ("dinov3-vitl16  sat493m", "300M", 0.783, "sat"),
        ("dinov2-with-registers-base", "86M", 0.766, "web"),
        ("dinov3-vit7b16  sat493m  (NF4)", "7B", 0.637, "sat"),
        ("dinov2-base", "86M", 0.621, "web"),
        ("dinov3-vit7b16  sat493m  (int8)", "7B", 0.618, "sat"),
        ("dinov3-vitl16  lvd1689m", "300M", 0.615, "web"),
        ("dinov3-vitb16  lvd1689m", "86M", 0.586, "web"),
        ("dinov2-with-registers-large", "300M", 0.545, "web"),
        ("dinov3-vit7b16  lvd1689m  (NF4)", "7B", 0.511, "web"),
        ("dinov3-vits16  lvd1689m", "22M", 0.181, "web"),
    ][::-1]
    fig, ax = plt.subplots(figsize=(10, 5.0))
    best = 0.783
    ax.axvspan(best - NOISE, best, color=BLUE, alpha=0.10, linewidth=0)
    ax.text(best - NOISE / 2, len(rows) - 0.42, "tied with the best\n(noise floor ±0.07)",
            ha="center", va="bottom", fontsize=9, color=INK2)
    for i, (name, params, v, kind) in enumerate(rows):
        ax.barh(i, v, height=0.6, color=BLUE if kind == "sat" else MUTED)
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{n}   ·   {p}" for n, p, _, _ in rows], color=INK)
    ax.set_xlim(0, 0.92)
    ax.set_ylim(-0.6, len(rows) + 0.5)
    ax.set_xlabel("AP@0.3")
    clean(ax, "x")
    ax.legend(handles=[Patch(color=BLUE, label="satellite pretraining (SAT-493M)"),
                       Patch(color=MUTED, label="web pretraining")],
              frameon=False, loc="lower left", bbox_to_anchor=(0, 0.955), ncol=2,
              fontsize=10, borderaxespad=0)
    fig.suptitle("An 86M model ties the best and beats both 7B checkpoints",
                 x=0.02, ha="left", fontsize=13.5, fontweight="bold", y=0.99)
    fig.text(0.02, -0.02,
             "Half b, --percentile 98 --min-patches 3, no negatives. dinov2-large is "
             "omitted: it returns exactly 0.000 from an unresolved bug.",
             fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, "backbones.png")


# ------------------------------------------------ 3. decision vs features
def levers():
    rows = [  # (label, delta AP@0.3, side)
        ("percentile + min-patches", +0.124, "d"),
        ("hard negatives", +0.092, "d"),
        ("local maxima, on top of negatives", +0.068, "d"),
        ("4-bit vs 8-bit quantisation (7B)", +0.019, "f"),
        ("best layer vs final layer", +0.013, "f"),
        ("attention keys vs tokens", -0.089, "f"),
        ("7B backbone vs 86M", -0.129, "f"),
    ][::-1]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.axvspan(-NOISE, NOISE, color=INK2, alpha=0.08, linewidth=0)
    ax.text(-NOISE + 0.004, len(rows) - 0.42, "noise floor ±0.07", ha="left",
            va="bottom", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.9)
    for i, (name, d, side) in enumerate(rows):
        ax.barh(i, d, height=0.58, color=BLUE if side == "d" else ORANGE)
        ax.text(d + (0.006 if d >= 0 else -0.006), i, f"{d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=10.5, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlim(-0.18, 0.18)
    ax.set_ylim(-0.6, len(rows) + 0.5)
    ax.set_xlabel("change in AP@0.3")
    clean(ax, "x")
    ax.legend(handles=[Patch(color=BLUE, label="change to the decision"),
                       Patch(color=ORANGE, label="change to the features")],
              frameon=False, loc="lower right", fontsize=10)
    fig.suptitle("Every gain came from the decision, none from the features",
                 x=0.02, ha="left", fontsize=13.5, fontweight="bold", y=0.99)
    fig.text(0.02, -0.02,
             "Each bar is one controlled comparison. The first two are measured on "
             "half b, local maxima, layer and keys on half a.",
             fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, "levers.png")


# --------------------------------------------------- 4. operating point
def operating_point():
    fig, ax = plt.subplots(figsize=(10, 2.9))
    rows = [("v2   --min-score 0.273", 120, 22, BLUE),
            ("v1   --min-score 0.413", 121, 83, MUTED)]
    for i, (name, tp, fp, c) in enumerate(rows):
        ax.barh(i + 0.19, tp, height=0.34, color=c)
        ax.barh(i - 0.19, fp, height=0.34, color=c, alpha=0.45)
        ax.text(tp + 2, i + 0.19, f"{tp} pools found", va="center", fontsize=10.5,
                fontweight="bold")
        ax.text(fp + 2, i - 0.19, f"{fp} false alarms", va="center", fontsize=10.5,
                fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlim(0, 160)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("detections kept after the threshold")
    clean(ax, "x")
    fig.suptitle("Same pools, 3.8× fewer false alarms  —  precision 0.59 → 0.85",
                 x=0.02, ha="left", fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.02, -0.07,
             "Half b, 146 objects. The v2 threshold was chosen on half a and applied "
             "unchanged.", fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, "operating_point.png")


if __name__ == "__main__":
    arc()
    backbones()
    levers()
    operating_point()
