#!/usr/bin/env python3
"""Merge several benchmark runs into one ranking.

    python merge_bench.py outputs/bench*/results.json
    python merge_bench.py outputs/bench/*/results.json --out outputs/all.md

benchmark.py always writes to the same folder, so a second run overwrites the
first. Give each run its own --out, then merge them here. Models measured
under DIFFERENT settings are flagged rather than silently ranked together:
comparing AP across percentiles is meaningless.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

KEYS = ["short", "AP@0.3", "AP@0.5", "R@1fp", "R@4fp", "precision", "recall",
        "sec_per_tile"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="results.json files or globs")
    p.add_argument("--out", default="", help="also write a markdown table")
    p.add_argument("--sort", default="AP@0.3")
    a = p.parse_args()

    paths = []
    for f in a.files:
        paths += [Path(x) for x in glob.glob(f)] or [Path(f)]

    rows, configs, seen = [], {}, {}
    for path in paths:
        if not path.exists():
            print(f"  skipped (missing): {path}")
            continue
        data = json.loads(path.read_text())
        cfg = data.get("config", {})
        sig = (cfg.get("percentile"), cfg.get("agg"), cfg.get("half"),
               cfg.get("n_tiles"), cfg.get("k_shot"),
               cfg.get("patches_per_object"))
        configs[sig] = configs.get(sig, 0) + 1
        for r in data.get("results", []):
            key = r["model"]
            if key in seen:
                print(f"  note: {r['short']} appears twice, keeping the first")
                continue
            seen[key] = True
            r["_sig"] = sig
            rows.append(r)
        for s in data.get("skipped", []):
            print(f"  skipped by {path.name}: {s['model']} - {s['reason'][:70]}")

    if not rows:
        print("nothing to merge")
        return 1

    if len(configs) > 1:
        print("\nWARNING: these runs did NOT use the same settings.")
        for sig, n in configs.items():
            print(f"  percentile={sig[0]} agg={sig[1]} half={sig[2]} "
                  f"tiles={sig[3]} k={sig[4]}  ({n} file(s))")
        print("  AP measured under different settings is not comparable.\n")

    rows.sort(key=lambda r: -r.get(a.sort, 0))
    w = max(len(r["short"]) for r in rows) + 2
    head = f"{'model':<{w}}" + "".join(f"{k:>13}" for k in KEYS[1:])
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['short']:<{w}}" + "".join(
            f"{r.get(k, 0):>13.3f}" if isinstance(r.get(k), (int, float))
            else f"{str(r.get(k, 0)):>13}" for k in KEYS[1:]))

    if a.out:
        L = ["| model | " + " | ".join(KEYS[1:]) + " |",
             "|" + "---|" * len(KEYS)]
        for r in rows:
            L.append("| `" + r["short"] + "` | " + " | ".join(
                f"{r.get(k, 0):.3f}" if isinstance(r.get(k), (int, float))
                else str(r.get(k, "")) for k in KEYS[1:]) + " |")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(L) + "\n")
        print(f"\nmarkdown -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
