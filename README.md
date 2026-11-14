# 🏊 pool-fewshot

### Teaching a computer to find swimming pools, by showing it five of them.

Welcome! This is a **research project**, which is a polite way of saying:
I try things, most of them fail, and I write down both.

The goal is not really pools. The goal is to **learn few-shot detection
properly and share everything with the community** — the code, the numbers,
and especially the mistakes, because those took the longest to produce.

Pools are just the perfect guinea pig: bright blue, easy to check by eye, and
nobody gets hurt when the model is wrong.

---

## What's in here

| folder | what it is | score (AP@0.3) |
|---|---|---|
| [`pool-fewshot-v1/`](pool-fewshot-v1) | the first version — a frozen model, 5 example crops, a cosine similarity | 0.783 |
| [`pool-fewshot-v2/pool-fewshot/`](pool-fewshot-v2/pool-fewshot) | the current version — hard negatives + local-maxima decoding | **0.890** |

**Start with v2.** It finds 140 of 146 pools on images it has never seen, with
no training at all. Its README has the figures, the tables and the commands.

---

## Things I learned that I did not expect 😅

- 🏠 **A white roof is a very convincing swimming pool** if you are a neural
  network. So is a pale terrace. v2 exists largely because of roofs.
- 🐘 **A 7-billion-parameter model lost to an 86-million one.** I made it fit
  on my GPU with great effort so it could score worse. Worth it for the story.
- 🧮 **For weeks, the code could not find more than 5 pools per image** — not
  because it couldn't see them, but because of a percentile. Some tiles have 8
  pools. It was arithmetic, not vision.
- 📄 **A published trick worth +10 points elsewhere made things worse here.**
  Papers are true where they were measured. Your data is somewhere else.
- 🤫 **One model still scores exactly 0.000** and I do not know why yet. It is
  in the README, not under the carpet.

---

## Why share all this?

Because when I started, the tutorials only showed what works. I learn more
from "here is what broke and how I noticed" — so that is what this repo tries
to be.

If you spot a mistake, have a better idea, or point this at something that is
not a pool (and it works, or collapses spectacularly): **open an issue, I
would love to hear it** ;)

*No pools were harmed during this research. Several roofs were wrongly accused.*