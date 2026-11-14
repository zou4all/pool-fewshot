# Backbone comparison — few-shot pool detection

- tiles: **102** (half `b`), **146** annotated objects
- support: **5 crops** from `/home/raze/AI_ECO/envirmnt_AI/new/pool-fewshot-v3/pool-fewshot/data/support5`
- settings: percentile 99.0, agg `mean`, 4 patches/object, object ~28 px

Only the backbone changes between rows. Everything else is identical, so the gap between two rows is attributable to the backbone.

## Results

| model | AP@0.3 | AP@0.5 | R@1fp | R@4fp | precision | recall | s/tile | notes |
|---|---|---|---|---|---|---|---|---|
| `dinov2-with-registers-base` | **0.674** | 0.471 | 0.630 | 0.664 | 0.10 | 0.85 | 0.12 | 86M, web, register tokens |
| `dinov3-vitl16-pretrain-sat493m` | **0.659** | 0.467 | 0.555 | 0.569 | 0.12 | 0.76 | 0.23 | 300M, SATELLITE pretraining |
| `dinov3-vitb16-pretrain-lvd1689m` | **0.657** | 0.397 | 0.541 | 0.582 | 0.17 | 0.78 | 0.11 | 86M, web, patch 16 |
| `dinov3-vitl16-pretrain-lvd1689m` | **0.560** | 0.331 | 0.548 | 0.582 | 0.14 | 0.77 | 0.85 | 300M, web, patch 16 |
| `dinov2-base` | **0.443** | 0.237 | 0.336 | 0.343 | 0.03 | 0.52 | 0.14 | 86M, web, patch 14 - the reference |
| `dinov2-large` | **0.000** | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | 0.29 | 300M, web - watch for fp16 overflow |

## Reading the table

- Best: **dinov2-with-registers-base**, AP@0.3 0.674.
- Spread between best and worst: +0.674 AP@0.3 — same code, same crops, same tiles.
- `R@4fp` is the operational figure: the share of objects found while allowing 4 false alarms per tile. AP alone collapses towards zero in a high-false-positive regime and stops discriminating.
- `precision` / `recall` are counted at IoU 0.3, which is the right threshold when the aim is *flagging a location* rather than segmenting it precisely.

---
Imagery © IGN — BD ORTHO®, Open Licence / Etalab.
