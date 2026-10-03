# 🏊 pool-fewshot

### Show it five swimming pools. It finds the rest.

No training. No fine-tuning. No labelled dataset.

You hand it five small crops of whatever you're looking for, point it at an
aerial photo it has never seen, and it draws boxes around the matches. That's
the whole idea — a frozen vision model, a prototype built from your five
examples, and a cosine similarity.

On 102 tiles it had never seen, five crops were enough to find **86% of the
pools**.

Pools aren't the goal, though. They're the test — easy enough that you can
tell at a glance when something is broken. Swap the five crops in
`data/support5/` and it looks for something else entirely. No code changes.

## 1. What it does

```
support crops (5 images)              target tile (512×512)
        │                                     │
        └──────────► frozen backbone ◄────────┘
                          │
              prototype   │   dense patch grid
                          ▼
                  cosine similarity
                          │
                     heatmap ──► percentile threshold
                                 ──► connected components
                                     ──► boxes + scores
                                         ──► score threshold
```

The pool is a **positive control**. Pools are easy — a colour threshold finds
most of them — and that is exactly the point: on a class where the right
answer is known, a broken pipeline is visible immediately. Once the chain is
validated, the same code points at a class where few-shot is actually
necessary (illegal waste dumps, damaged roofs, unregistered structures) by
swapping the support folder. Nothing else changes.

---



### Where this started

The **EcoAI** project
([zou4all/EcoAI-Intelligent-Innovations-for-a-Greener-Future](https://github.com/zou4all/EcoAI-Intelligent-Innovations-for-a-Greener-Future))
detects swimming pools imagery the standard way: a **YOLOv5m**
detector fine-tuned on an annotated dataset of Cannes tiles — 883 training
images, 255 validation, one class .

It is a built supervised pipeline, and it is the reason this repository
exists with the dataset.. 

### What is different here

| | EcoAI (YOLOv5) | pool-fewshot |
|---|---|---|
| examples needed | 883 annotated images | **5 crops** |
| training | fine-tuning, GPU-hours | **none** |
| change the target class | re-annotate, re-train | swap a folder |
| backbone | trained weights | frozen, off the shelf |




### Why few-shot, and why pools

The class I actually want to detect is **illegal waste dumps** (*dépôts
sauvages*) — and there is no public dataset of 883 annotated ones. There may
be five photographs. Supervised detection has nothing to work with; the
problem is not that the model is weak, it is that the training set cannot be
built. That is the regime few-shot learning exists for, and it is why I wanted
to learn it properly rather than read about it.

So pools are a **positive control**, not the goal. They are easy — a colour
threshold finds most of them — and that is exactly the point. On a class where
the right answer is already known, a broken pipeline shows up immediately.


That ordering was deliberate: validate the method on a class you can verify,
then move to the class you care about.

### What I was actually trying to learn

Writing the detector was the easy half. The half that took the time, and
taught the most:

- An evaluation that does not lie to you. Splitting by source image, ranking
  support crops on a different half from the one you report on, global AP
  instead of per-tile averages,Most were found *after* producing a number that looked fine.
- Knowing which lever actually moves. Two settings (`--percentile`,
  `--min-patches`) were worth more than any change of backbone: **+0.192
  AP@0.5 and +0.172 R@4fp**, no retraining. I would not have guessed that, and
  I only know it because the experiment was cheap enough to run.
- Reading a result instead of accepting it. `AP = 0.000` meant "truncated
  list" once and "collapsed heatmap" another time, never "bad model". The
  fastest diagnostic in this whole project was looking at the annotated images.

### Learning in the open

I am learning this in public, and the repository is written that way: the
mistakes are documented next to the fixes, the confounded comparison is
labelled as confounded , and the one unresolved bug is described rather
than hidden. A clean README that only shows what worked would have been
quicker to write and far less useful to me later, and to anyone else
starting from the same place.

So: if you spot something wrong here, please tell me ;) I would much rather
know. Issues, corrections and better ideas are all welcome, and if you point
this at a different class and it works (or falls apart), I would love to hear
how it went.

---


## 2. Results

102 evaluation tiles (half `b`), 146 annotated objects, 5 support crops,
`--percentile 98 --min-patches 3`. Only the backbone changes between rows.

| model | AP@0.3 | AP@0.5 | R@4fp | s/tile | params |
|---|---|---|---|---|---|
| `dinov3-vitl16-pretrain-sat493m` | **0.783** | **0.659** | 0.740 | 0.24 | 300M |
| `dinov2-with-registers-base` | 0.766 | 0.556 | **0.760** | **0.13** | 86M |
| `dinov2-base` | 0.621 | 0.392 | 0.479 | 0.13 | 86M |
| `dinov3-vitl16-pretrain-lvd1689m` | 0.615 | 0.286 | 0.541 | 0.86 | 300M |
| `dinov3-vitb16-pretrain-lvd1689m` | 0.586 | 0.325 | 0.555 | 0.09 | 86M |
| `dinov2-large` | — | — | — | 0.29 | 300M |

`dinov2-large` returns exactly 0.000 and is **a bug, not a measurement** — see
Known limitations. It is excluded from every claim below.

### Four effects, measured separately

**1. Register tokens — +0.145 AP@0.3, +0.281 R@4fp.**
`dinov2-base` → `dinov2-with-registers-base` is the cleanest controlled
comparison here: same architecture, same size, same pretraining data, one
difference. Recall at the operating point goes **0.479 → 0.760**, a 59%
relative gain. Darcet et al. showed that ViTs allocate high-norm "artifact"
tokens in low-information patches, and that register tokens absorb them. This
pipeline uses *only* dense patch features, so it feels that repair directly —
far more than a classification benchmark would.

**2. Satellite pretraining beats web pretraining — +0.168 AP@0.3 at equal size.**
`dinov3-vitl16-sat493m` 0.783 vs `dinov3-vitl16` (LVD-1689M web) 0.615, same
300M architecture. This runs against the DINOv3 authors' general finding that
the web checkpoint wins on detection benchmarks. On overhead imagery, on this
task, it loses. Worth knowing before assuming a leaderboard transfers to your
domain.

**3. Bigger is not better when features are frozen.**
The 86M `dinov2-with-registers-base` beats both 300M web models by a wide
margin and is 2–7× faster. It also isn't gated on HuggingFace. For deployment
it is arguably the better pick; `sat493m` wins only where box *tightness*
matters (AP@0.5 0.659 vs 0.556).

**4. Which support crops you pick — ×500.**
Measured under the older settings, on the 84-tile set:

| support | AP@0.3 |
|---|---|
| 30 worst crops | 0.001 |
| 87 random crops | 0.098 |
| 20 best | 0.210 |
| 12 best | 0.309 |
| 8 best | 0.426 |
| **5 best** | **0.554** |

Fewer, better crops beat more, worse crops — decisively. The obvious
explanation was blur; it was wrong. Sharpness correlates **−0.38** with final
AP, crop *size* correlates **+0.44**. Small crops are mostly coping and
surround with very little water, and they teach the prototype the wrong thing.

Caveat: this curve was measured at `--percentile 99 --min-patches 1`, before
the tuning below. A diffuse prototype is penalised hardest when the threshold
budget is tight, so the effect is probably overstated at the current settings.
Re-measuring it — with `--agg knn`, which is the right pairing for a large
support set — is open work.

### Two settings were worth more than any model change

Same model (`sat493m`), same crops, same tiles:

| configuration | AP@0.5 | AP@0.3 | R@4fp | missed |
|---|---|---|---|---|
| `--percentile 99 --min-patches 1` (defaults) | 0.467 | 0.659 | 0.568 | 35 |
| `--percentile 99 --min-patches 3` | 0.492 | 0.656 | 0.568 | 41 |
| **`--percentile 98 --min-patches 3`** | **0.659** | **0.783** | **0.740** | **20** |

**+0.192 AP@0.5 and +0.172 R@4fp from two flags.** No retraining, no new
model. The percentile sweep explains why:

| percentile | squares kept (of 4225) | tp | fp | missed | AP@0.3 |
|---|---|---|---|---|---|
| 99 | 42 | 105 | 278 | 41 | 0.656 |
| **98** | **85** | 126 | 546 | **20** | **0.783** |
| 97 | 127 | 128 | 780 | 18 | 0.780 |
| 96 | 169 | 128 | 975 | 18 | 0.776 |

A pool covers about 4×4 = 16 squares. At percentile 99 the tile keeps only 42
squares — enough for two or three pools. Tiles in this dataset hold a median
of 2 objects and up to 8, so **a third pool could not be returned no matter
how brightly it lit up**. Half the misses were arithmetic, not perception.
Below 98 recall saturates while false positives keep doubling.

---

## 3. Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# torch first, matched to your hardware
pip install torch --index-url https://download.pytorch.org/whl/cu121
# RTX 50xx (Blackwell) needs cu128 — earlier builds have no sm_120 kernels

pip install -r requirements.txt
```

DINOv3 weights are gated. Accept the licence on the model page, then
`hf auth login`.

Check the maths without network or model — 17 assertions on geometry, IoU,
label round-trip, tiling, AP, FROC and the colour proposals:

```bash
python run.py selftest
```

Every script also accepts `--model raw-pixels`, a control encoder built from
16×16 downscaled RGB. No weights, no download, no GPU. If the real backbone
does not clearly beat it, the pipeline is measuring colour, not features.

---

## 4. Data preparation — from YOLO labels to what the model actually eats

This is the part that is easy to get wrong, because **the model never sees a
YOLO box.** Labels are an annotation format; the backbone consumes pixels
resized to a very specific scale. Two conversions sit between them.

### 4.1 Where the data comes from

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
measured only where pools exist:

```bash
python run.py download --lat 43.518066 --lon 5.461946 --negatives-only
```

→ 204 tiles: 120 annotated, 84 empty. That is the set every number in §2 was
measured on.

Three things worth knowing about this import:

- **The dataset is YOLO, not COCO.** `--yolo` reads an
  `images/<split>/` + `labels/<split>/` tree; `--json` / `--images` is for
  COCO releases and cannot read this one.
- **`--short` is not cosmetic.** It turns
  `CANNES_TILES_512x512-1000_png.rf.8d2e5c...` into `cannes-1000`. `source_of()`
  splits names on `_` and would otherwise read part of the Roboflow hash as a
  tile offset, silently corrupting the `--half` split.
- **Two images are 424×512.** Every image is opened at its own size so their
  boxes are correct, but a single `--object-px` cannot fit every tile exactly.

### 4.2 The label file

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

### 4.3 Labels serve two completely different purposes

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

### 4.4 Support crop → prototype bank

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

### 4.5 Tile → patch grid

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
   │  CL2N centring (subtract the mean, re-normalise)
   ▼
heatmap 65×65  →  percentile  →  connected components  →  boxes
   │
   └─ box coordinates scaled back ×(512/1040) to tile pixels, then
      compared to the ground-truth boxes from 4.2 by IoU
```

### 4.6 The one constraint that ties it together

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
from the grid arithmetic in §2.

On this dataset it suggests `--object-px 45 --percentile 94`. Apply that **as a
pair**: raising `object_px` coarsens the grid (46×46 instead of 65×65), which
is why the matching percentile is lower. The measured results in §2 use the
default `object_px` (~28) with `--percentile 98` — a different, tested
operating point. Mixing the two settings gives neither.

### 4.7 Two rules for the files themselves

- A tile and its label share the exact stem: `cannes-1000.jpg` ↔
  `cannes-1000.txt`. A tile with no `.txt` is a negative — which is correct
  and useful, so long as it really contains no object.
- Do not rename tiles cut from a mosaic. The `_<ox>_<oy>` suffix is what
  `source_of()` reads to group tiles by source image, and that grouping is
  what keeps `--half a` and `--half b` disjoint. Rename them and the leak
  comes back silently.

---

## 5. Full run, step by step

### Step 3 — rank and pick the support crops

```bash
python rank_support.py --half a --top 30
python pick_support.py --n 5 --from data/candidates --to data/support5 --sheet
```

`rank_support.py` scores every annotated object as a solo 1-shot exemplar and
sorts them. `--sheet` writes a contact sheet so you can *look* at what you are
about to feed the model — which is how the crop-size effect was found.

### Step 4 — evaluate

```bash
python run_dense.py --support-dir data/support5 --half b \
                    --percentile 98 --min-patches 3
```

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
python benchmark.py --phase eval --models dinov3
python benchmark.py --phase predict --best facebook/dinov2-with-registers-base
```

### Step 6 — see what the model is doing

```bash
python show_results.py --half b --n 12 --heatmap \
                       --percentile 98 --min-patches 3
```

Green = correct, red = false alarm, blue dashed = missed object. A number
tells you how well it works; this tells you why.

---

## 6. Prediction and thresholding

The percentile is a **budget**, not a decision: it always keeps the top N% of
squares, so a tile with no pool at all still produces boxes. For deployment
you want an absolute cut-off. That is `--min-score`.

### 6.1 Choose the threshold from data, not from one image

`detections.csv` already records every detection with its score and its
TP/FP/MISS label, so sweeping thresholds costs nothing — no GPU, no second
pass:

```bash
python sweep_threshold.py outputs/bench/predictions/detections.csv
```

Measured on 102 tiles, `sat493m`, `--percentile 98 --min-patches 3`:

| `--min-score` | pools kept | false alarms | precision | recall |
|---|---|---|---|---|
| none (0.328) | 126 / 126 | 409 | 0.24 | 0.86 |
| **0.405** | 123 | 123 | 0.50 | **0.84** |
| **0.413** | 121 | 83 | 0.59 | 0.83 |
| 0.430 | 104 | 29 | 0.78 | 0.71 |
| **0.438** | 97 | **10** | **0.91** | 0.66 |
| 0.464 | 38 | 0 | 1.00 | 0.26 |

Two sensible operating points:

- **Assisted triage** (a human checks the shortlist) → `--min-score 0.413`.
  121 of 126 pools kept, false alarms 409 → 83. **5× fewer images to open for
  4% of recall.**
- **High confidence** (automated output) → `--min-score 0.438`. 97 pools, 10
  false alarms, precision 0.91.

```bash
python predict.py --images data/tiles --support-dir data/support5 \
       --model facebook/dinov3-vitl16-pretrain-sat493m \
       --percentile 98 --min-patches 3 --min-score 0.413 --sheet
```

Or let the code derive it from a false-alarm budget:

```bash
python benchmark.py --calibrate --fpr 4     # → threshold 0.3282 for ≤4 fp/tile
```

### 6.2 Why a global threshold is meaningful here

Boxes are *placed* by the centred heatmap (the tile's own background is
subtracted, so objects stand out) but *scored* by the raw one. Without
centring the scores stay comparable across tiles — which is what makes 0.44 on
one tile mean the same as 0.44 on another.

The dataset's 22 negative tiles confirm it empirically. They produce 96 false
alarms, 4.4 per tile — but their **maximum score is 0.437**, below the 0.455
median of true pools. At `--min-score 0.438`, all 96 disappear. On ground that
genuinely contains no pool, the model never crosses a clean ceiling.

### 6.3 What survives, and why it is interesting

At 0.438 the ~10 remaining false alarms sit **entirely on tiles that already
contain pools**, several scoring *above* the median of true positives. That is
not the profile of a confusion; it is the profile of **missing annotations**.
Before treating them as errors, look at them — some are probably real pools
the labeller skipped, in which case the true precision is higher than reported.

Conversely, the weakest true positives are almost all *small* pools
(16–24 m²), a direct consequence of `--object-px`. A threshold kills those
first, so any recall figure quoted with a threshold is a recall figure about
medium and large pools.

### 6.4 The methodological trap

Choosing the threshold on half `b` and then reporting on half `b` is the same
selection bias that `--half` exists to prevent. For a defensible number: run
prediction on `--half a`, pick the threshold there, then apply it unchanged to
`b`.

---

## 7. Methodology — the parts that are easy to get wrong

Most of these were found the hard way, by a result that looked fine and wasn't.

**Split by source image, not by tile.** Tiles cut from a mosaic overlap by
64 px, so the same pool appears in several of them. A tile-level split puts one
object on both sides and inflates everything. `source_of()` groups tiles by
source and `split_half()` assigns whole sources to half `a` or half `b`.

**Rank on one half, evaluate on the other.** This one was caught by a question
rather than a test: `rank_support.py` was scoring candidates on 24 tiles that
were inside the evaluation set. The crops were then selected *because* they
scored well there. That is selection leakage, and it silently inflates the
headline number.

**Global AP, not the mean of per-tile APs.** A tile with no object necessarily
scores 0, so averaging per-tile APs divides the result by the number of
negatives. It measures the proportion of empty tiles, not the detector.

**FROC as well as AP.** At high false-positive rates AP collapses towards zero
and stops discriminating. `R@4fp` — recall while allowing 4 false alarms per
tile — stays informative, and answers the question an operator actually asks.
It is also the metric that separates a real gain from cosmetics: tuning
`--min-patches` left it at 0.568 unchanged, while the percentile moved it to
0.740. Only one of those was finding pools.

**`max_detections` must not truncate.** With a few thousand windows per tile
the one true positive can land around rank 145. Cutting the list at 50 throws
it away and prints `AP = 0.000`, which reads like a broken model rather than a
truncated list. The default here is 200.

**CL2N centring.** Subtract the distribution mean, then re-normalise
(SimpleShot, Wang et al. 2019). In high dimension a few "hub" vectors sit
close to everything and win every comparison regardless of content. Two lines
of code, several points of AP.

**Rotations.** Each support crop is encoded at 4 rotations × 2 mirrors,
otherwise the prototype encodes the orientation of your examples.

**fp16 can return NaN.** `dinov3-vitl16` overflows in half precision. NaN ≥
threshold is `False` everywhere, the mask comes out empty, no box is produced,
and AP prints exactly `0.000` — with no error. `DenseEncoder` checks
`torch.isfinite` and falls back to fp32.

**Patch size is read, never assumed.** DINOv2 uses 14, DINOv3 uses 16, SigLIP
keeps `patch_size` inside `vision_config` rather than at the config root. A
wrong value shifts the grid with no exception raised, and every result is
quietly wrong.

**Episodes, with a standard deviation.** `--episodes 10` repeats the run over
random draws of the k support crops. A single number from a single draw is not
a result, and a large standard deviation is not noise to hide — it quantifies
how sensitive the method is to the choice of examples.

---

## 8. Known limitations

- **`dinov2-large` returns exactly 0.000** and it is not the fp16 NaN bug —
  `torch.isfinite` passes. The likely cause is a collapsed heatmap: if the
  similarity map is near-constant, the percentile lands on the max and the
  mask becomes degenerate. Unresolved; the model is excluded from all claims.
- **The sliding-window comparison is confounded.** The baseline scored
  AP@0.3 0.022 against 0.783 for the dense path, but it ran with *random*
  support while the dense path used curated crops. The raw gap therefore
  mixes two changes. `python run.py experiment --support-dir data/support5`
  isolates the mechanism; it takes about an hour and has not been run.
- **Single scale.** Objects far from `--object-m` are penalised, and small
  pools are the systematic casualty. A multi-scale pass would help and is not
  implemented.
- **`--calibrate` is ignored when `--best` is passed** in phase 2, because the
  threshold is only read from `results.json` on the auto-selection path.
- **146 objects** is a small evaluation set. Differences under ~0.05 AP should
  not be over-read.
- **One geographic area.** Nothing here shows the numbers hold over a
  different region, season or flight campaign.
- **Ground truth is imperfect.** §6.3 gives direct evidence of missed
  annotations. Reported precision is a lower bound.

---

## 8.1. Roadmap — this is v1

Everything above works and every number is measured. But this is a **first
version**, written while learning the method, and it stops in several obvious
places. Here is what I know is missing.

### Backbones I haven't tried yet

Satellite pretraining gave the single biggest model-side gain (+0.168 AP@0.3
at equal size, §3), so the highest-signal direction is **more remote-sensing
backbones**, not bigger general ones:

- **Geospatial foundation models** — Prithvi (IBM/NASA), Clay, SatMAE,
  Scale-MAE. Built for overhead imagery, which is exactly where `sat493m`
  beat its web-trained twin. Check current ids on the Hub; some need a loader
  of their own rather than plain `AutoModel`.
- **SigLIP2** (`google/siglip2-base-patch16-224`) — already supported in the
  code (no CLS token, `patch_size` in `vision_config`), never benchmarked. A
  contrastive image-text objective fails differently from a self-supervised
  one, and "differently" is the interesting part.
- **AIMv2** (Apple) — autoregressive image pretraining, a third family
  entirely.
- **CLIP** — as a reference point more than a candidate.

`benchmark.py` takes any HuggingFace id, so adding one is a command, not a
patch:

```bash
python benchmark.py --models google/siglip2-base-patch16-224 \
       --percentile 98 --min-patches 3
```

### Method changes worth measuring

- **Multi-scale.** Everything runs at one scale, and small pools are the
  systematic casualty — the weakest true positives are almost all 16–24 m²
  . This is probably the largest remaining gain.
- **`--object-px 45`.** The scale is currently set to ~28 px while the real
  median is 45. Fixing it changes the grid, so it has to be re-tuned as a pair
  with `--percentile` (§5.6).
- **`--agg knn` with a larger support set.** The "5 beats 20" curve was
  measured under the old throttled settings, with `mean` aggregation that
  dilutes a diverse prototype across 160 vectors. That result deserves
  re-measuring.
- **Negative-aware prototypes.** The surviving false alarms sit on roofs and
  bright terraces — a structured confusion, not noise. Subtracting a
  "negative" prototype built from those could be cheap and effective.
- **Re-tune `--min-patches` at percentile 98.** It was tuned at 99; the two
  interact.


### Known bugs still open

- `dinov2-large` returns exactly 0.000. Running it in fp32 on CPU for 3098
  seconds gave the same result, so the precision hypothesis is dead — it is a
  collapsed heatmap, cause unknown.
- `--calibrate` is ignored when `--best` is passed to phase 2.
- Nothing warns you when CUDA silently falls back to CPU. I found out after a
  four-hour benchmark that should have taken three minutes.


**Contributions and corrections are welcome** on any of the above. If one of
these is something you already know the answer to, I would rather learn it
from you than rediscover it slowly.

---


## 9. Layout

```
run.py              selftest · download · label · check · ceiling · experiment
run_dense.py        dense evaluation (the method that works)
benchmark.py        compare every backbone, then run the winner
predict.py          inference on new imagery → JPEG, CSV, GeoJSON
show_results.py     green / red / blue-dashed visualisation
rank_support.py     rank candidate support crops
pick_support.py     copy the top-N into a support folder
sweep_threshold.py  choose --min-score from detections.csv, no GPU
import_eco.py       import an external COCO-style dataset

src/geo.py          tile ↔ lat/lon, GeoJSON export
src/data.py         boxes, IoU, NMS, WMTS download, tiling, colour proposals
src/dense.py        DenseEncoder, prototype, heatmap, boxes  ← the core
src/fewshot.py      CL2N, prototypes, control encoder
src/sliding.py      sliding-window baseline (kept as a control)
src/metrics.py      global AP, FROC, counts, episodes
src/annotate.py     drawing, contact sheets, heatmap overlays

data/raw            WMTS tile cache
data/tiles          evaluation tiles
data/labels         YOLO labels (cls cx cy w h, normalised)
data/candidates     ranked support candidates
data/support5       the 5 chosen support crops
outputs/            results, reports, annotated images
```

### Recommended settings

```bash
--model facebook/dinov3-vitl16-pretrain-sat493m   # or dinov2-with-registers-base
--percentile 98
--min-patches 3
--min-score 0.413                                  # assisted triage
```

---

## 10. References

- Snell, Swersky, Zemel — *Prototypical Networks for Few-shot Learning*, 2017
- Wang et al. — *SimpleShot: Revisiting Nearest-Neighbor Classification*, 2019
  (CL2N centring)
- Oquab et al. — *DINOv2: Learning Robust Visual Features without
  Supervision*, 2023
- Darcet et al. — *Vision Transformers Need Registers*, 2023
- Siméoni et al. — *DINOv3*, 2025 (Gram anchoring, register tokens, the
  `sat493m` satellite checkpoint)

---
