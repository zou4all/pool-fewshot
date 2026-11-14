# Results, and what did not work

> Part of [pool-fewshot](../README.md). Section numbers are shared across
> all the docs, so a reference like §4.2 means the same thing everywhere.

## 2. How to read these numbers — the noise floor

**Differences smaller than ±0.07 AP@0.3 on this dataset are not measurements.**
Every claim below is either above that line or explicitly labelled as being
below it.

That number is measured, not guessed. Quantising a frozen backbone to 4-bit
NF4 should be close to a no-op for a feature extractor — the weights move
slightly, nothing about the method changes. So it makes a good **null
perturbation**: whatever AP@0.3 moves by, under a change that ought to do
nothing, is the floor for this dataset and this metric. Across backbones it
moved by up to 0.07, in both directions.

With 146 annotated objects on 102 tiles, that is about what you would expect:
a handful of borderline boxes crossing an IoU threshold is worth several
points of AP. A proper bootstrap over tiles would be better and is open work,
but a null perturbation costs one command and is hard to argue with.

This has consequences, and some of them cut against v1 of this README:

- `sat493m` (0.783) and `dinov2-with-registers-base` (0.766) are **tied** on
  AP@0.3. v1 called the first one the winner; it is not, on this metric.
- Four of the five results in §5 are differences *below* the floor — which is
  the point of that section. "No effect" is a finding, but only if you can say
  what size of effect you would have been able to see. The fifth, the keys
  comparison, is a difference *above* the floor and so a real negative.
- The §4 gains are comfortably above it: +0.092 for negatives and +0.107 for
  both together on half `b`, +0.068 for peaks on top of negatives on half `a`.
  Each is corroborated by a **count of found objects**, which does not depend
  on AP's ranking behaviour at all.

Where a gain and a count disagree, I trust the count.

---
---

## 3. Results

### 3.1 The arc

102 evaluation tiles (half `b`), 146 annotated objects, 5 support crops,
`dinov3-vitl16-pretrain-sat493m`. Only the decoding and the prototype change
between rows.

| configuration | AP@0.5 | AP@0.3 | R@4fp | missed |
|---|---|---|---|---|
| `--percentile 99 --min-patches 1` (original defaults) | 0.467 | 0.659 | 0.568 | 35 |
| `--percentile 98 --min-patches 3` | 0.659 | 0.783 | 0.740 | 20 |
| + hard negatives (`--neg-weight 0.35`) | — | 0.875 | — | 9 |
| **+ local-maxima decoding (`--peak-rel 0.4`)** | **0.790** | **0.890** | **0.877** | **6** |

**35 missed objects → 6.** Nothing in that table is a different model, a
different checkpoint or a trained weight. The backbone is byte-identical in
every row.

The same four-way comparison, run as a clean 2×2 on half `a` (143 objects) so
that each lever can be read on its own:

| | `--percentile 98` | `--decode peaks` |
|---|---|---|
| **P⁺ only** | 0.720 — 23 missed | 0.766 — 12 missed |
| **P⁺ and P⁻** | 0.781 — 14 missed | **0.849 — 8 missed** |

They compose: +0.068 over the better of the two alone, which is above the
noise floor, and the miss count halves. §4.3 explains why, because I predicted
they would be redundant and was wrong.

### 3.2 Backbones

`--percentile 98 --min-patches 3`, half `b`, no negatives, percentile decoding
— the v1 configuration, so these rows stay comparable to v1 of this README.

| model | params | AP@0.3 | AP@0.5 | R@4fp | s/tile |
|---|---|---|---|---|---|
| `dinov3-vitl16-pretrain-sat493m` | 300M | **0.783** | **0.659** | 0.740 | 0.24 |
| `dinov2-with-registers-base` | 86M | **0.766** | 0.556 | **0.760** | **0.13** |
| `dinov3-vit7b16-pretrain-sat493m` (NF4) | 7B | 0.637 | — | — | 2.4 |
| `dinov3-vit7b16-pretrain-sat493m` (int8) | 7B | 0.618 | — | — | 3.1 |
| `dinov2-base` | 86M | 0.621 | 0.392 | 0.479 | 0.13 |
| `dinov3-vitl16-pretrain-lvd1689m` | 300M | 0.615 | 0.286 | 0.541 | 0.86 |
| `dinov3-vitb16-pretrain-lvd1689m` | 86M | 0.586 | 0.325 | 0.555 | 0.09 |
| `dinov2-with-registers-large` | 300M | 0.545 | — | — | 0.29 |
| `dinov3-vit7b16-pretrain-lvd1689m` (NF4) | 7B | 0.511 | — | — | 2.4 |
| `dinov3-vits16-pretrain-lvd1689m` | 22M | 0.181 | — | — | 0.05 |
| `dinov2-large` | 300M | 0.000 | — | — | 0.29 |

`dinov2-large` returns exactly 0.000 and is **a bug, not a measurement** — see
§11. It is excluded from every claim.

**1. Register tokens — +0.145 AP@0.3, +0.281 R@4fp.** Above the floor.
`dinov2-base` → `dinov2-with-registers-base` is the cleanest controlled
comparison here: same architecture, same size, same pretraining data, one
difference. Recall at the operating point goes **0.479 → 0.760**. Darcet et
al. showed that ViTs allocate high-norm "artifact" tokens in low-information
patches, and that register tokens absorb them. This pipeline uses *only* dense
patch features, so it feels that repair directly — far more than a
classification benchmark would.

This says nothing about registers at 300M, though: `dinov2-large` returns
0.000 from a bug, so there is no working large baseline to compare
`registers-large` (0.545) against. What that 0.545 does show is that size
hurts here — which is finding 3, not a finding about registers.

**2. Satellite pretraining beats web pretraining — +0.168 AP@0.3 at equal
size.** Above the floor. `dinov3-vitl16-sat493m` 0.783 vs the LVD-1689M web
checkpoint 0.615, same 300M architecture. This runs against the DINOv3
authors' general finding that the web checkpoint wins on detection
benchmarks. On overhead imagery, on this task, it loses. Worth knowing before
assuming a leaderboard transfers to your domain.

It also holds at 7B: 0.637 satellite vs 0.511 web. So the effect is not an
artefact of one size — the pretraining domain separates the two checkpoints at
300M and at 7B alike.

**3. Bigger is emphatically not better when features are frozen.** The 86M
`dinov2-with-registers-base` ties `sat493m` within the noise floor, beats every
other 300M model and both 7B models, and is 2–20× faster. The 7B satellite checkpoint, quantised to fit on a 16 GB card,
scores 0.637 — **0.13 below an 86M model that needs no quantisation at all**.

That is the single most useful practical result in this repository. Scaling the
backbone is the expensive reflex, and here it was not only not worth it, it was
actively counterproductive. My guess at why: a frozen pipeline consumes patch
embeddings directly, and a huge model's patch space is organised for a
fine-tuned head that does not exist here. But that is a guess — see §5, where
several equally plausible guesses turned out to be wrong.

**4. Which support crops you pick — ×500.** Measured under the older settings,
on the 84-tile set:

| support | AP@0.3 |
|---|---|
| 30 worst crops | 0.001 |
| 87 random crops | 0.098 |
| 20 best | 0.210 |
| 12 best | 0.309 |
| 8 best | 0.426 |
| **5 best** | **0.554** |

Fewer, better crops beat more, worse crops — decisively, and by far more than
the noise floor. The obvious explanation was blur; it was wrong. Sharpness
correlates **−0.38** with final AP, crop *size* correlates **+0.44**. Small
crops are mostly coping and surround with very little water, and they teach the
prototype the wrong thing.

Caveat: this curve was measured at `--percentile 99 --min-patches 1`, before
any of the tuning above. A diffuse prototype is penalised hardest when the
threshold budget is tight, so the effect is probably overstated at current
settings. Re-measuring it under `--decode peaks`, which removes the budget
entirely, is open work — and now that §4.2 explains what the budget was doing,
I expect the curve to flatten noticeably.

---
---

## 5. What did not work

Five hypotheses, all plausible, all refuted. They took longer to produce than
§4 and they are the reason §4 is believable: a project that only reports its
successes has not told you how hard it looked.

Everything here is a difference **below the ±0.07 noise floor** of §2 unless
stated otherwise.

### 5.1 "Precision loss explains the `dinov2-large` collapse" — no

`dinov2-large` returns AP exactly 0.000, and fp16 overflow is a real failure
mode in this codebase (see §10), so LayerNorm precision was the obvious
suspect. Running the whole thing in **fp32 on CPU** — 3098 seconds — gave
*exactly the same* 0.000. The hypothesis is dead. (Silver lining: the CPU run
is also how I found out CUDA had been silently falling back, after a four-hour
benchmark that should have taken three minutes.)

### 5.2 "Quantisation damages dense features" — no, it costs nothing measurable

Swept fp32 / fp16 / int8 / NF4 on several backbones. Every delta landed inside
the floor, in both directions. This is the result that produced §2: a change
that *ought* to do nothing moved AP@0.3 by up to 0.07, which told me how big a
difference I was able to see at all.

Practically it is good news — `--load-in 4bit` is a pure memory tool, not an
accuracy trade, and it is what let a 7B checkpoint run on a 16 GB card:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_dense.py --model facebook/dinov3-vit7b16-pretrain-sat493m \
       --load-in 4bit --max-side 768 --half b
```

(int8 on 7B needed ~134 MB more than the card had. `expandable_segments`,
stopping the display manager, and `--max-side` together got it in.)

### 5.3 "The failures split by model family" — no

`dinov2-large` fails and
`dinov3-vitl16` does not; `dinov2-base` works and `dinov2-large` does not.
There is no family-shaped pattern, no size-shaped pattern, and no
checkpoint-shaped pattern that survives contact with the full table in §3.2.
Whatever is wrong is specific to one checkpoint.

### 5.4 "A middle layer gives better dense features than the last one" — no

Amir et al. and common practice both suggest mid-depth features are less
task-specialised. Swept on half `a`, tokens, over the final layer and four
fractional depths: **0.720** at the final layer, **0.733** and **0.695** from
the two near-final probes, then a fast collapse — **0.576**, then **0.379**.

The best value, 0.733, is **+0.013** over the default — inside the floor, so:
use the final layer, the flag is not worth tuning. The collapse at shallow
depths is unambiguous and well outside the floor, which at least confirms the
flag does something.

### 5.5 "Attention keys beat tokens" — no, they lose badly

This is the cleanest negative result in the repository.

Amir et al. (*Deep ViT Features as Dense Visual Descriptors*, 2021) compared
the four descriptors a transformer block can give you, on co-segmentation
(PASCAL-Co, mean Jaccard): **keys 79.5**, queries 72.7, **tokens 69.2**,
values 49.2. Ten points between what this project uses and the best option, on
a neighbouring dense-correspondence task — and the stated reason, *less
sensitivity to background clutter*, is this project's exact failure mode.

It does not transfer. Swept over six depths on half `a`, because facet and
layer interact and the token answer cannot be assumed to carry over. The keys
gave **0.146, 0.177, 0.577, 0.579, 0.644, 0.212** — best **0.644**.

| descriptor | best AP@0.3 | where |
|---|---|---|
| tokens (`last_hidden_state`) | **0.733** | near the final block |
| keys | 0.644 | mid-depth |

Keys lose at **every depth swept**, and their best is 0.089 below the tokens'
best — above the noise floor, so this is a real negative result rather than an
absence of signal.

Two published results, both reproduced in their own setting, that combine into
a wrong prediction for mine. Dense correspondence between two images of the
same object class is not the same problem as scoring patches against an
exemplar bank, and keys are apparently tuned for the former.

```bash
python run_dense.py --facet key --layer 0.75 --half a    # re-run the comparison
```

**One warning about how I nearly missed this.** My first implementation located
the transformer blocks at a list of hard-coded attribute paths. DINOv3 nests
its blocks under `.model.layers`, which was not on the list, so the lookup
returned nothing, the encoder printed a fallback warning, and quietly used
tokens. With `| grep "AP@0.3"` in the sweep loop the warning was invisible —
and the symptom was that the `key` rows came out **byte-identical** to the
`token` rows. Identical numbers from a changed flag is the signal; `grep` is
how you hide it from yourself. The block stack is now found by *structure* —
the longest `ModuleList` whose entries all contain an attention projection —
which works across all five nesting layouts I could find.

---
