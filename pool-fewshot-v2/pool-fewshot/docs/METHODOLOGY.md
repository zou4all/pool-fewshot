# Methodology, limitations and roadmap

> Part of [pool-fewshot](../README.md). Section numbers are shared across
> all the docs, so a reference like §4.2 means the same thing everywhere.

## 10. Methodology — the parts that are easy to get wrong

Most of these were found the hard way, by a result that looked fine and wasn't.

**Measure your noise floor before anything else.** §2. I measured mine tenth,
after forming and discarding several hypotheses whose evidence I could not
properly weigh at the time. A null perturbation — a change that ought to do
nothing — costs one command.

**Split by source image, not by tile.** Tiles cut from a mosaic overlap by
64 px, so the same pool appears in several of them. A tile-level split puts one
object on both sides and inflates everything. `source_of()` groups tiles by
source and `split_half()` assigns whole sources to half `a` or half `b`.

**Every hyperparameter is chosen on `a` and confirmed on `b`.** Support crops,
negative weight, peak-rel, percentile, min-patches, min-score. See §9.1 for
the one I got wrong, and for how an honest protocol also gives you a stronger
result — "the threshold transfers" is a claim you can only make this way.

**Global AP, not the mean of per-tile APs.** A tile with no object necessarily
scores 0, so averaging per-tile APs divides the result by the number of
negatives. It measures the proportion of empty tiles, not the detector.

**FROC as well as AP.** At high false-positive rates AP collapses towards zero
and stops discriminating. `R@4fp` — recall while allowing 4 false alarms per
tile — stays informative, and answers the question an operator actually asks.
It is also what separates a real gain from cosmetics: `--min-patches` left it
unchanged at 0.568 while the percentile moved it to 0.740, and it is the metric
that showed the negatives were re-ranking rather than just filtering (§4.3).

**Watch the object counts, not only the AP.** `tp` / `fp` / `miss` do not
depend on AP's ranking behaviour, and when they disagree with the AP the count
is the one to trust. Every §4 gain is corroborated by a miss count.

**Identical numbers from a changed flag mean the flag did nothing.** Not that
the change does not matter. §5 ends with the hour I spent believing keys were
equivalent to tokens, when in fact the hook was never attached.

**Never pipe a sweep through `grep`.** A fallback warning, a CUDA-fell-back-to-
CPU message and a NaN notice are all exactly the lines `grep "AP@0.3"` throws
away. Run one configuration bare, read all of it, *then* loop.

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

**Find model internals by structure, not by name.** The block stack lives at a
different attribute path in every architecture, and a hard-coded list of paths
fails silently the moment a new one appears (§5). Walking the module tree for
the longest `ModuleList` whose entries all contain an attention projection
works everywhere I have tested it.

**Episodes, with a standard deviation.** `--episodes 10` repeats the run over
random draws of the k support crops. A single number from a single draw is not
a result, and a large standard deviation is not noise to hide — it quantifies
how sensitive the method is to the choice of examples.

---
---

## 11. Known limitations

- **`dinov2-large` returns exactly 0.000** and it is not the fp16 NaN bug —
  `torch.isfinite` passes, and fp32 on CPU gives the same result (§5.1). The
  remaining hypothesis is a collapsed heatmap: if the similarity map is
  near-constant, the percentile lands on the max and the mask becomes
  degenerate. `--decode peaks` would be immune to that failure mode and has
  not been tried on this checkpoint — the cheapest open experiment in the
  repository.
- **146 objects, ±0.07 AP@0.3 noise floor.** The hard ceiling on everything
  here. A proper bootstrap CI over tiles, and more evaluation tiles, would both
  help.
- **One geographic area, one flight campaign, one season.** Nothing here shows
  the numbers hold elsewhere. This is the limitation I would attack first if the
  goal were a product rather than a method.
- **The sliding-window comparison is confounded.** The baseline scored
  AP@0.3 0.022 against 0.783 for the dense path, but it ran with *random*
  support while the dense path used curated crops. The raw gap therefore
  mixes two changes. `python run.py experiment --support-dir data/support5`
  isolates the mechanism; it takes about an hour and has not been run.
- **Single scale.** Objects far from `--object-m` are penalised, and small
  pools are the systematic casualty. A multi-scale pass would help and is not
  implemented.
- **1673 false alarms before thresholding** under `--decode peaks`. Harmless
  given the ranking (§4.3) and removed by `--min-score`, but it means the
  pre-threshold precision number is not meaningful on its own.
- **`--calibrate` is ignored when `--best` is passed** in phase 2, because the
  threshold is only read from `results.json` on the auto-selection path.
- **Ground truth is imperfect.** §9.4 gives direct evidence of missed
  annotations. Reported precision is a lower bound.
- **Nothing warns you when CUDA silently falls back to CPU.** I found out after
  a four-hour benchmark that should have taken three minutes.

---
---

## 12. Roadmap — this is v2

v1 stopped at "two flags beat any model change". v2 stops at "everything that
worked was on the decision side". The natural next question is whether that is
a fact about this method or a fact about how little I have tried on the feature
side — so the plan is split accordingly.

### Finish the decision side, since that is where everything has worked

- **`--decode peaks` on `dinov2-large`.** One command, and it tests the
  collapsed-heatmap hypothesis directly (§11). This is the first thing to run.
- **Re-measure the support-set curve under `--decode peaks`.** The ×500 effect
  in §3.2 was measured with a tight budget, which is exactly the condition a
  diffuse prototype suffers most under. I expect it to flatten; if it does not,
  that is more interesting still.
- **Iterate the negative mining.** One round of mining gave +0.092. A second
  round, mining against the *improved* detector, is the obvious next step — and
  the classic failure mode (the bank drifts towards the target class) is
  something this setup can actually detect, because the empty tiles stay empty.
- **Multi-scale.** Still the largest structural gap, and small pools are still
  the systematic casualty.
- **`--object-px 45`,** re-tuned as a pair with the decoder settings (§7.6).

### Keep probing the feature side, but expect little

Four attempts, four nulls — so the prior is now low, and the designs should be
cheap:

- **Geospatial foundation models** — Prithvi (IBM/NASA), Clay, SatMAE,
  Scale-MAE. Satellite pretraining is the one feature-side change that *did*
  pay (+0.168, §3.2), so remote-sensing backbones remain the best bet. Some
  need a loader of their own rather than plain `AutoModel`.
- **SigLIP2** (`google/siglip2-base-patch16-224`) — already supported in the
  code (no CLS token, `patch_size` in `vision_config`), never benchmarked. A
  contrastive image-text objective fails differently from a self-supervised
  one, and "differently" is the interesting part.
- **AIMv2** (Apple) — autoregressive image pretraining, a third family
  entirely.
- **Queries and values**, for completeness, now that `--facet` works. Amir et
  al. rank queries second; if queries also lose, the finding is "this task
  wants block outputs", which is worth stating cleanly.

### Method changes that would be a different project

- **Semi-supervised self-training.** The negatives are already mined without
  labels; pseudo-labelling the confident *positives* and re-building the
  prototype from them is the same trick in the other direction.
- **A learned 2-layer head** on the frozen features, trained on the 5 crops
  plus the mined negatives. This stops being training-free, which is the point
  of the repository — but it is the honest comparison to make before claiming
  training-free is enough.

**Contributions and corrections are welcome** on any of the above. If one of
these is something you already know the answer to, I would rather learn it from
you than rediscover it slowly.

---
