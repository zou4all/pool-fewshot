#!/usr/bin/env python3
"""Fix --facet: find the transformer blocks by structure, not by name.

    python patch_facet_fix.py

WHY THIS IS NEEDED. The first version looked for the block stack at a list of
hard-coded paths (encoder.layer, blocks, layers...). DINOv3 in transformers
nests its blocks under `.model.layers`, which was not in that list, so
find_blocks returned [], the encoder printed a fallback warning and quietly
used tokens. With `| grep AP@0.3` in the loop the warning was invisible, and
the "key" rows came out byte-identical to the "token" rows.

THE FIX. Stop guessing paths. Walk every ModuleList in the model and keep the
longest one whose entries all contain an attention projection. That is the
block stack in any ViT, however it is nested. Verified against five layouts:
.model.layers, .encoder.layer, .blocks, a three-level nesting, and a model
carrying a 50-entry decoy ModuleList with no attention in it.

Run from the project root, after patch_facet.py. Idempotent; writes nothing
if the pattern does not match.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

OLD = '''BLOCK_PATHS = ("encoder.layer", "encoder.layers", "blocks", "layer", "layers")


def find_blocks(net):
    """The list of transformer blocks, or [] when the layout is unknown."""
    for path in BLOCK_PATHS:
        obj, ok = net, True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            return list(obj)
    return []'''

NEW = '''ATTN_LEAVES = ("key", "query", "value", "k_proj", "q_proj", "v_proj",
               "to_k", "to_q", "to_v", "qkv", "in_proj", "qkv_proj")


def _has_attention(block, nn) -> bool:
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in ATTN_LEAVES:
            return True
    return False


def find_blocks(net):
    """The transformer block stack, found by STRUCTURE rather than by name.

    Hard-coded paths break the moment an architecture nests differently.
    DINOv3 puts its blocks under `.model.layers`, which an earlier version of
    this function did not look for: it returned [], the encoder fell back to
    tokens, and the whole facet comparison measured nothing.

    So: walk every ModuleList and keep the longest one whose entries ALL
    contain an attention projection. That is the block stack in any ViT,
    whatever the nesting, and a decoy ModuleList of plain Linears cannot be
    mistaken for it.
    """
    import torch
    nn = torch.nn
    best = []
    for _name, mod in net.named_modules():
        if not isinstance(mod, (nn.ModuleList, nn.Sequential)) or len(mod) < 2:
            continue
        if len(mod) > len(best) and all(_has_attention(b, nn) for b in mod):
            best = list(mod)
    return best'''


def main() -> int:
    d = ROOT / "src" / "dense.py"
    if not d.exists():
        sys.exit(f"{d} not found - run this from the project root")
    t = d.read_text()

    if "_has_attention" in t:
        print("  dense.py: already fixed")
        return 0
    if "def find_blocks" not in t:
        sys.exit("ABORTED: --facet is not installed. Run patch_facet.py first.")
    if t.count(OLD) != 1:
        sys.exit(f"ABORTED, nothing written: found {t.count(OLD)} matches, "
                 "expected 1. Send me src/dense.py and I will adjust.")

    d.write_text(t.replace(OLD, NEW, 1))
    print("  dense.py: block finder replaced")
    print("\nCheck it works BEFORE the sweep - without grep, so you can see it:\n")
    print("  python run_dense.py --model facebook/dinov3-vitl16-pretrain-sat493m \\")
    print("         --support-dir data/support5 --half a --percentile 98 \\")
    print("         --min-patches 3 --facet key --quiet | head -5")
    print("\nYou want a line like:")
    print("  [facet] key <- block 24/24, attention.qkv (fused qkv)")
    print("A 'falling back to tokens' line means it is still not hooked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())