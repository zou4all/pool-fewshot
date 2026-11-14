# Method: the two additions, the data, and thresholding

> Part of [pool-fewshot](../README.md). Section numbers are shared across
> all the docs, so a reference like §4.2 means the same thing everywhere.

## 4. The two methods that made the difference

Neither trains anything. Neither needs a new annotation. Both act on what
happens *after* the backbone has spoken.

### 4.1 Hard negatives — teaching it what it keeps getting wrong

**The problem.** A prototype is a description of what you are looking for: the
mean of your example vignettes, and a cosine similarity against it. That is a
*generative* approach — it models the target class and ignores everything else.

Here that description became too good for its own good. Recall reached 86%,
but precision sat at 24%: bright roofs and terraces answer "yes" to the
question *does this look like a pool?* The confusion is not noise, it is
**structured** — the same kinds of object, every time.

**The fix.** Add a second question, and subtract it:

```
s(x) = cos(x, P+) - lambda * max cos(x, P-)
```

The score no longer says only "this resembles the target", but "this resembles
the target **and not** what usually fools me".

`max`, not `mean`. A patch is suspect if it resembles *any one* of the
negatives. A mean would let one atypical negative dilute precisely the
confusion the bank exists to catch — and with 80 vectors in the bank, that
dilution is severe.

**Where the negatives come from — this is the semi-supervised part.** From
tiles known to contain no pool. Run the detector on them unchanged: **every
box it returns is false by construction.** The highest-scoring ones are the
hardest, and those are exactly the ones worth keeping. No new annotation is
created; the only label used is "this tile is empty", which you already have.

```bash
python mine_negatives.py --half a --top 40 --sheet
python run_dense.py --half b --neg-dir data/negatives --neg-weight 0.35
```

`--sheet` writes a contact sheet of what was mined. Look at it. Mine were
almost entirely white roof sections and pale stone terraces, which is a useful
confirmation that the bank captured the real confusion rather than random
background.

**Mined on one half, evaluated on the other.** Mining on the half you report
on would mean selecting negatives *because* they work there — the measured gain
would be its own selection coming back around. Same rule as `--half`, same
reason.

**The surprise.** The prediction was "fewer false alarms, same number of
pools". The opposite happened: **11 more pools found**, false alarms barely
moved. Subtracting negatives did not clean up the output — it **freed up
room**. Which is where the second method comes from.

### 4.2 Local-maxima decoding — a percentile is not a threshold

**The problem.** The heatmap has to become boxes. The original method kept the
squares above a **percentile** and grouped neighbours into connected
components. That looks like a threshold. It only looks like one: a percentile
keeps a **fixed share of the grid**, whatever is actually on it.

```
grid 65 x 65          = 4225 squares
percentile 98         ->   84 squares kept
one object covers 4x4 =   16 squares
                      ----------------------
ceiling               ->    5 objects per tile
```

Tiles in this dataset hold up to **eight** pools. From the sixth onwards,
**none can be returned**, however brightly it lights up in the map. That was
never a failure of perception. It was arithmetic.

It also explains §4.1's surprise: while the budget was the binding constraint,
subtracting negatives mostly served to free squares for real objects, so the
gain showed up in recall instead of precision.

**The fix.** Decide per object rather than per grid. Every local maximum of the
heatmap is one candidate; two objects on a tile no longer compete, and a tile
holding eight returns eight.

```bash
python run_dense.py --decode peaks --peak-rel 0.4 --half b
```

**What stops noise from producing a peak per bump** is not a threshold on the
map — that would be a budget again. It is a **grown region** around each peak
with a minimum size: a noise spike dies within a cell or two, a real object
holds its value across a dozen or more. That is `--min-patches`, now applied
to the grown region rather than to a connected component.

**The trap that cost me two attempts.** The growth threshold must be relative
to the **contrast**, not to the peak's value. Cosine similarities live in a
narrow band well above zero — background ≈ 0.37, a pool ≈ 0.45. So a
multiplicative threshold like `0.6 × peak` lands *below the background*, the
flood fills the whole tile, and every tile returns exactly `max_boxes` boxes.
The correct form is:

```
grow while  hm >= floor + rel * (peak - floor)
```

Symptom to recognise: a constant box count per tile, identical on empty tiles
and crowded ones. That is a decoder that has stopped reading the map.

**The knobs.**

| flag | what it does | swept to |
|---|---|---|
| `--peak-rel` | how far below the peak a box grows, so its size. This is what AP@**0.5** responds to. | **0.4** (0.6 → 0.721, 0.8 → 0.519) |
| `--peak-floor` | percentile below which peaks are ignored. Deliberately loose. | 90 (default) |
| `--peak-window` | neighbourhood for the local max; 0 derives it from `--patches-per-object` | 0 (default) |

At `--peak-rel 0.8` the boxes are too tight and miss the IoU threshold — 57
missed objects. At 0.4, 12. The map was the same in both runs; only the box
geometry changed. Worth remembering before blaming a backbone for poor AP@0.5.

### 4.3 Why they compose

My prediction was that they would be redundant: if the negatives' gain came
from freeing budget, then removing the budget should make the negatives
pointless. The 2×2 in §3.1 says otherwise, and the reason is that the two
levers act on **different axes**:

- **decoding** controls *how many* candidates can come out of a tile;
- **negatives** control *which* candidates rank at the top.

While the budget was binding, those two were entangled, and the negatives'
contribution was partly spent on the budget. Remove the budget and their
contribution becomes **pure re-ranking**, and the miss count on half `a` shows
it: peaks alone leave 12 objects unfound, peaks with negatives leave 8, at
`R@4fp` 0.839.

That is also what makes 1673 false alarms tolerable. They pile up at the
*bottom* of the ranking. Given a budget of four false alarms per tile, you
still recover 88% of the pools.

---
---

## 7. Data preparation — from YOLO labels to what the model actually eats

This is the part that is easy to get wrong, because **the model never sees a
YOLO box.** Labels are an annotation format; the backbone consumes pixels
resized to a very specific scale. Two conversions sit between them.

### 7.1 Where the data comes from

The evaluation set is the pool dataset of the **EcoAI** project
([zou4all/EcoAI-Intelligent-Innovations-for-a-Greener-Future](https://github.com/zou4all/EcoAI-Intelligent-Innovations-for-a-Greener-Future)):
a Roboflow export of aerial tiles over Cannes, already annotated in YOLO
format — 883 train + 255 val images of 512×512, one class (`pool`), 2564
boxes in the training split.

```bash
cd ~/AI_ECO/envirmnt_AI
git clone --depth 1 \
  https://github.com/zou4all/EcoAI-Intelligent-Innovations-for-a-Greener-Future.git eco

cd pool-fewshot
python import_eco.py --yolo ../eco/EcoAI-CaseStudy/EcoAI-V1.0/dataset \
                     --splits train --limit 120 --short --prefix "" --reset
python run.py check
```

which reproduces exactly:

```
tiles            : 120
annotated tiles  : 120
objects          : 289
tile size (px)   : 512x512 (118), 424x512 (2)
object size (px) : median 45  p10 26  p90 65
objects / tile   : median 2  max 8
```

The dataset contains **no negative tiles** — every image has at least one
pool. Empty ground is added separately from IGN, otherwise precision is
measured only where pools exist, and there is nothing to mine negatives from:

```bash
python run.py download --lat 43.518066 --lon 5.461946 --negatives-only
```

→ 204 tiles: 120 annotated, 84 empty. That is the set every number above was
measured on. Those 84 empty tiles do double duty — they are what makes
precision meaningful, and they are the raw material for §4.1.

Three things worth knowing about this import:

- **The dataset is YOLO, not COCO.** `--yolo` reads an
  `images/<split>/` + `labels/<split>/` tree; `--json` / `--images` is for
  COCO releases and cannot read this one.
- **`--short` is not cosmetic.** It turns
  `CANNES_TILES_512x512-1000_png.rf.8d2e5c...` into `cannes-1000`.
  `source_of()` splits names on `_` and would otherwise read part of the
  Roboflow hash as a tile offset, silently corrupting the `--half` split. (The
  regex also has to survive the `512x512` in the middle of the name, which the
  first version did not.)
- **Two images are 424×512.** Every image is opened at its own size so their
  boxes are correct, but a single `--object-px` cannot fit every tile exactly.

### 7.2 The label file

`data/labels/cannes-1000.txt`, one line per object, normalised to [0, 1]:

```
0 0.661 0.339 0.071 0.107
│   │     │     │     └── height / tile height
│   │     │     └──────── width / tile width
│   │     └────────────── centre y / tile height
│   └──────────────────── centre x / tile width
└──────────────────────── class (always 0 here)
```

Normalised coordinates survive resizing, which is why the format is used. But
they are meaningless without the image, so `load_labels()` always takes the
real width and height — read from the image, never assumed.

### 7.3 Labels serve two completely different purposes

```
data/labels/cannes-1000.txt
        │
        ├─► ×512 → Box(320, 146, 357, 201)
        │          GROUND TRUTH, compared to detections by IoU
        │
        └─► +25% margin → crop 46×68 px → data/candidates/*.jpg
                   SUPPORT CROP, the few-shot example itself
```

The 25% margin is deliberate. A pool cropped edge to edge is a rectangle of
water with no coping, no terrace, no surroundings — and that is measurably a
worse exemplar (crop size correlates +0.44 with final AP).

### 7.4 Support crop → prototype bank

```
crop 46×68 px
   │  resize so the object spans exactly 4 patches
   │    DINOv2 : 4 × 14 = 56×56 px
   │    DINOv3 : 4 × 16 = 64×64 px
   │
   │  × 4 rotations (0°, 90°, 180°, 270°) × 2 mirrors = 8 views
   │    seen from above, an object has no canonical orientation —
   │    without this the prototype encodes the orientation of your examples
   │
   │  encode each view → mean-pool its patches → 1 vector
   ▼
5 crops × 8 views = bank of 40 vectors (1024-dim for ViT-L)
```

The negative bank is built the same way from the mined crops, so the two are
directly comparable in the same space — 40 crops × 2 views = 80 vectors.

### 7.5 Tile → patch grid

```
data/tiles/cannes-1000.jpg   512×512
   │
   │  resize ×2.03 → 1040×1040
   │    so that a pool has the SAME APPARENT SIZE as in the support crops
   │
   │  chop into 16-px squares → 65 × 65 = 4225 vectors
   ▼
grid (65, 65, 1024)
   │
   │  cosine similarity against the 40-vector bank
   │  minus λ × max similarity against the 80-vector negative bank   (§4.1)
   │  CL2N centring (subtract the mean, re-normalise)
   ▼
heatmap 65×65  →  percentile OR local maxima  →  grow  →  boxes     (§4.2)
   │
   └─ box coordinates scaled back ×(512/1040) to tile pixels, then
      compared to the ground-truth boxes from 7.3 by IoU
```

### 7.6 The one constraint that ties it together

**A patch must see the same amount of object on the support side and on the
target side.** That is the entire job of `--object-px`:

```
scale = (patches_per_object × patch_size) / object_px
```

Set `--object-px 28` when your pools are really 45 px, and the support crop is
built at 4 patches per object while the tile presents them at 6.5. You are
comparing a close-up to a distant view, and similarity collapses. It fails
silently — no error, just a lower score.

Get the true value from your own data:

```bash
python run.py check
```

which reads the real tile size from the images and prints the median object
size in pixels, plus a suggested `--object-px` / `--percentile` pair computed
from the grid arithmetic in §4.2.

On this dataset it suggests `--object-px 45 --percentile 94`. Apply that **as a
pair**: raising `object_px` coarsens the grid (46×46 instead of 65×65), which
is why the matching percentile is lower. The measured results above use the
default `object_px` (~28) with `--percentile 98` — a different, tested
operating point. Mixing the two settings gives neither.

### 7.7 Two rules for the files themselves

- A tile and its label share the exact stem: `cannes-1000.jpg` ↔
  `cannes-1000.txt`. A tile with no `.txt` is a negative — which is correct
  and useful, so long as it really contains no object.
- Do not rename tiles cut from a mosaic. The `_<ox>_<oy>` suffix is what
  `source_of()` reads to group tiles by source image, and that grouping is
  what keeps `--half a` and `--half b` disjoint. Rename them and the leak
  comes back silently.

---
---

## 8. Full run, step by step

### Step 1 — get the data

See §7.1. You need both halves: the annotated tiles and the empty ones.

### Step 2 — rank and pick the support crops

```bash
python rank_support.py --half a --top 30
python pick_support.py --n 5 --from data/candidates --to data/support5 --sheet
```

`rank_support.py` scores every annotated object as a solo 1-shot exemplar and
sorts them. `--sheet` writes a contact sheet so you can *look* at what you are
about to feed the model — which is how the crop-size effect was found.

### Step 3 — mine the hard negatives

```bash
python mine_negatives.py --half a --top 40 --sheet
```

Runs the detector on the label-less tiles of half `a`. Every box is false by
construction; the 40 highest-scoring crops become `data/negatives/`. Open the
sheet and check they look like your real confusion (for me: white roofs, pale
terraces).

### Step 4 — evaluate

```bash
python run_dense.py --support-dir data/support5 --half b \
                    --model facebook/dinov3-vitl16-pretrain-sat493m \
                    --min-patches 3 \
                    --neg-dir data/negatives --neg-weight 0.35 \
                    --decode peaks --peak-rel 0.4
```

That is the full v2 configuration — the last row of §3.1.

### Step 5 — compare every backbone, then run the winner

```bash
python benchmark.py --percentile 98 --min-patches 3 --calibrate --fpr 4
```

Phase 1 runs all models on identical inputs → `outputs/bench/report.md`,
`results.csv`, `results.json`.
Phase 2 takes the highest AP@0.3 and writes annotated images →
`outputs/bench/predictions/`, `results_sheet.jpg`.

```bash
python benchmark.py --models fast          # two models, quick pass
python benchmark.py --phase eval --models dinov3 --out outputs/bench_dinov3
python benchmark.py --phase predict --best facebook/dinov2-with-registers-base
```

**Always pass `--out` when running groups separately** — otherwise each run
overwrites the previous one's `results.json`, and `merge_bench.py` exists
because I learned that the slow way:

```bash
python merge_bench.py outputs/bench_* --out outputs/bench_all
```

### Step 6 — see what the model is doing

```bash
python show_results.py --half b --n 12 --heatmap \
                       --neg-dir data/negatives --neg-weight 0.35 \
                       --decode peaks --peak-rel 0.4
```

Green = correct, red = false alarm, blue dashed = missed object. A number
tells you how well it works; this tells you why. `--heatmap` also writes the
similarity map next to each result, which is the only way to tell "the object
was not detected" from "the object was detected and the decoder dropped it" —
a distinction that turned out to matter a great deal (§4.2).

---
---

## 9. Prediction and thresholding

The percentile and the peak decoder both produce a *ranked candidate list*,
not a decision: an empty tile still yields boxes. For deployment you want an
absolute cut-off. That is `--min-score`.

### 9.1 Choose the threshold on one half, apply it to the other

`detections.csv` records every detection with its score and its TP/FP/MISS
label, so sweeping thresholds costs nothing — no GPU, no second pass. Which
makes it dangerously easy to sweep on the half you then report on.

**The right order**, and the one the numbers below use:

```bash
# 1. predict on half a, sweep there
python benchmark.py --phase predict --half a \
       --best facebook/dinov3-vitl16-pretrain-sat493m \
       --neg-dir data/negatives --neg-weight 0.35 --decode peaks --peak-rel 0.4
python sweep_threshold.py outputs/bench/predictions/detections.csv
#    -> --min-score 0.273

# 2. apply that number, unchanged, to half b
python predict.py --images data/tiles --half b --support-dir data/support5 \
       --model facebook/dinov3-vitl16-pretrain-sat493m \
       --neg-dir data/negatives --neg-weight 0.35 \
       --decode peaks --peak-rel 0.4 --min-score 0.273 --sheet
```

Half `b`'s own optimum is **0.280**. The threshold chosen blind on `a` lands
within 0.007 of it, which is the evidence that it transfers — and it is
evidence you only get by doing it in this order.

I got this wrong first: I swept the negative weight on half `b` and reported on
half `b`. Same bias, one level up. Any hyperparameter chosen on the reported
half is a hyperparameter fitted to it, and `--neg-weight`, `--peak-rel`,
`--percentile`, `--min-patches` and `--min-score` were all re-chosen on `a`
after that.

### 9.2 The operating point

Half `b`, 102 tiles, 146 objects, threshold 0.273 from half `a`:

| | pools kept | false alarms | precision | recall |
|---|---|---|---|---|
| v1 (`--min-score 0.413`) | 121 | 83 | 0.59 | 0.83 |
| **v2 (`--min-score 0.273`)** | **120** | **22** | **0.85** | **0.82** |

**Same pools, 3.8× fewer false alarms.** And an operating point that did not
previously exist at all: 133 pools for 44 false alarms, **91% recall** — where
the v1 configuration topped out at 86% with *no* threshold whatsoever.

The two thresholds are not comparable as numbers (0.413 vs 0.273): subtracting
the negative term shifts the whole score range down. Always re-sweep after
changing the scoring function.

### 9.3 Why a global threshold is meaningful here

Boxes are *placed* by the centred heatmap (the tile's own background is
subtracted, so objects stand out) but *scored* by the raw one. Because the
score never has the tile's own background subtracted from it, it stays
comparable across tiles — which is what makes 0.27 on one tile mean the same
as 0.27 on another.

The empty tiles confirm it empirically: on ground that genuinely contains no
pool, the maximum score sits below the median of true pools, so a single
threshold removes essentially all of them at once.

### 9.4 What survives, and why it is interesting

The remaining false alarms at a high threshold sit **entirely on tiles that
already contain pools**, several scoring *above* the median of true positives.
That is not the profile of a confusion; it is the profile of **missing
annotations**. Before treating them as errors, look at them — some are probably
real pools the labeller skipped, in which case true precision is higher than
reported.

Conversely, the weakest true positives are almost all *small* pools
(16–24 m²), a direct consequence of `--object-px`. A threshold kills those
first, so any recall figure quoted with a threshold is a recall figure about
medium and large pools.

---
