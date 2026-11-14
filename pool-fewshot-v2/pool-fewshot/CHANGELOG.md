# Changelog

## [2.0.0] — November 2026

Two additions, neither of which trains anything or needs a new annotation:
a bank of mined **hard negatives**, and **local-maxima decoding** in place of
the percentile. Same backbone, same five support crops, same evaluation tiles.

| half `b`, 102 tiles, 146 objects | v1.0.0 | **v2.0.0** |
|---|---|---|
| AP@0.3 | 0.783 | **0.890** |
| AP@0.5 | 0.659 | **0.790** |
| R@4fp | 0.740 | **0.877** |
| missed objects | 20 | **6** |
| at the operating point | 121 pools / 83 false alarms | **120 / 22** |
| precision there | 0.59 | **0.85** |

**3.8× fewer false alarms for the same pools.** And an operating point that did
not exist in v1 at all: 133 pools for 44 false alarms, 91% recall, where v1
topped out at 86% with no threshold whatsoever.

The backbone is byte-identical between those two columns. Nothing was
retrained, re-annotated or swapped.

---

### Added

**Hard negative mining** — §4.1 of the README.
`mine_negatives.py`, plus `--neg-dir` and `--neg-weight` on the evaluation
scripts. Runs the detector on tiles known to be empty, where every box it
returns is false by construction, and keeps the highest-scoring crops as a
negative bank. The score becomes:

```
s(x) = cos(x, P+) - lambda * max cos(x, P-)
```

`max`, not `mean`: a patch is suspect if it resembles *any one* negative, and a
mean would let one atypical negative dilute the very confusion the bank exists
to catch. The only label used is "this tile is empty", which the dataset
already provided — this is the semi-supervised part of the method.

Worth +0.092 AP@0.3 and 11 more pools found. The gain arrived as **recall**,
not precision, which is what led to the second addition.

**Local-maxima decoding** — §4.2 of the README.
`--decode peaks`, with `--peak-rel`, `--peak-floor` and `--peak-window`.

A percentile is not a threshold, it is a **budget**: it keeps a fixed share of
the grid whatever is on it. At `--percentile 98` on a 65×65 grid that is 84
squares, and an object covers about 16, so **no tile could ever return more
than five objects.** Tiles in this dataset hold up to eight. Half the misses
were arithmetic, not perception — and it is also why subtracting negatives
raised recall: it was freeing budget rather than filtering.

Every local maximum is now one candidate, judged in its own window. Noise is
rejected by growing a region around each peak and requiring a minimum size,
not by a threshold on the map, which would reintroduce the budget.

**Facet extraction** — `--facet token|key|query|value`.
Reads the attention keys, queries or values of a chosen block instead of its
output tokens. Added to test Amir et al. (2021); the result is negative (see
below), but the flag stays because the comparison is worth being able to
re-run, and the block-finding code it needed is useful on its own.

**Layer selection** — `--layer`, taking the final layer, a negative index or a
fraction of the depth.

**Quantisation** — `--load-in 8bit|4bit` and `--max-side`.
What makes the 7B checkpoints fit on a 16 GB card. It turned out to cost
nothing measurable, so it is a memory tool rather than an accuracy trade.

**`merge_bench.py`** — merges several `--out` benchmark runs into one report.
Needed because running model groups separately silently overwrote
`results.json` each time.

**`.gitignore`** — the repository was tracking 346 generated files, 35 MB of
annotated JPEGs and `.pyc`. `data/support5/` and `data/negatives/` stay
tracked: they are *inputs* to the numbers above, and regenerating them needs
gated weights and a GPU.

**A noise floor** — §2 of the README, and the most important addition.
Quantising a frozen backbone to NF4 ought to be a no-op, so whatever AP@0.3
moves by under that change is the floor for this dataset. It moved by up to
**±0.07**, in both directions. Differences smaller than that are not
measurements, and the README now labels every claim accordingly.

---

### Changed

- **Recommended settings** are now `--neg-dir data/negatives --neg-weight 0.35
  --decode peaks --peak-rel 0.4 --min-patches 3 --min-score 0.273`.
- **`--min-score` 0.413 → 0.273.** Not a tightening: subtracting the negative
  term shifts the whole score range down, so the two numbers are not
  comparable. Always re-sweep after changing the scoring function.
- **`sat493m` is no longer described as the best backbone.** Against the new
  noise floor, its 0.783 and `dinov2-with-registers-base`'s 0.766 are **tied**
  on AP@0.3. v1 called the first one the winner; that claim did not survive
  knowing how large a difference was visible. `registers-base` is now the
  recommended deployment pick — 2× faster, 3.5× smaller, not gated.
- **`--min-patches` now applies to the grown region** around a peak rather
  than to a connected component.
- **The threshold protocol is now stated and followed.** Every hyperparameter
  — support crops, negative weight, peak-rel, percentile, min-patches,
  min-score — is chosen on half `a` and confirmed on half `b`. Half `b`'s own
  optimum threshold is 0.280 against the 0.273 chosen blind on `a`, and that
  agreement is the evidence the threshold transfers.
- **`sweep_threshold.py`** translated to English, in line with the rest of the
  project.
- **README** restructured: the noise floor comes first because it governs how
  everything else reads, the two new methods get their own section, and
  §5 collects what failed.

---

### Fixed

- **`--facet key` silently did nothing.** The block stack was located at a list
  of hard-coded attribute paths; DINOv3 nests its blocks under `.model.layers`,
  which was not on the list, so the lookup returned nothing and the encoder
  quietly fell back to tokens. The warning it printed was being swallowed by
  `| grep "AP@0.3"` in the sweep loop. Detected because the `key` rows came out
  **byte-identical** to the `token` rows. The stack is now found by structure —
  the longest `ModuleList` whose entries all contain an attention projection —
  verified against five nesting layouts.
- **The peak decoder returned `max_boxes` boxes on every tile.** The growth
  threshold was a fraction of the peak's *value*; cosine similarities sit in a
  narrow band well above zero (background ≈ 0.37, pool ≈ 0.45), so
  `0.6 × peak` landed below the background and the flood filled the tile. Now
  relative to the **contrast**: `floor + rel × (peak − floor)`.
- **`sweep_threshold.py` marked every improving row** as "best F1" instead of
  only the winner, which read as several recommendations.
- **A selection leak in the negative-weight sweep.** `--neg-weight` was swept
  on half `b` and then reported on half `b` — the same bias `--half` exists to
  prevent, one level up. Re-chosen on `a`.
- **`import_eco.py --short`** mangled `CANNES_TILES_512x512-1000`: the `512x512`
  in the middle of the name broke the regex, and a corrupted stem silently
  corrupts the `--half` split.
- **`import_eco.py`** gained `--yolo`. The EcoAI dataset is a Roboflow YOLO
  export, not COCO, and the COCO reader could not open it.

---

### Measured and rejected

Five plausible hypotheses, all refuted, all below the ±0.07 floor unless
noted. They took longer to produce than everything above, and they are why §4
is believable.

| hypothesis | verdict |
|---|---|
| precision loss explains the `dinov2-large` collapse | **no** — fp32 on CPU gives the same exact 0.000 |
| quantisation damages dense features | **no** — every delta inside the floor |
| the failures split by model family | **no** — no family, size or checkpoint pattern survives |
| a middle layer beats the final one | **no** — best is +0.013, inside the floor |
| attention keys beat tokens (published +10 pts) | **no** — 0.644 vs 0.733, *above* the floor, so a real negative |

The pattern across them is the one genuine finding of this release:
**every attempt to improve the features failed, and every change to the
decision worked.** A 7B backbone scored 0.637 against an 86M model's 0.766 —
80× the parameters, and worse.

---

## [1.0.0] — Initial release

Frozen-backbone few-shot detection: a prototype built from five support crops,
a dense patch grid, cosine similarity, CL2N centring, a percentile threshold
and connected components. Evaluation split by source image, global AP and
FROC, episodic evaluation, a raw-pixel control encoder, and a benchmark across
nine backbones.

Headline: AP@0.3 0.783 on 102 unseen tiles from five crops, with
`--percentile 98 --min-patches 3`.

The two findings that held up, and still hold against the v2 noise floor:

- **Register tokens** are worth +0.145 AP@0.3 and +0.281 R@4fp on an otherwise
  identical architecture.
- **Satellite pretraining beats web pretraining** by +0.168 AP@0.3 at equal
  size, against the DINOv3 authors' general finding — and it holds at 7B too.
