# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze expert-tier activation histograms for routing skew.

Reads the `<path>.rank{r}.json` files dumped by
--expert-tier-activation-hist-path and reports, per layer and overall:
how many experts cover 50/90/99% of routed activations, and the mass
lost by pruning the coldest N experts. Hash-routed layers (0-2) are
reported separately: they are exact table lookups and exempt from
pruning.
"""

import argparse
import glob
import json

import numpy as np

HASH_LAYERS = {0, 1, 2}


def coverage(counts: np.ndarray, frac: float) -> int:
    """Experts needed (hottest first) to cover `frac` of the mass."""
    s = np.sort(counts)[::-1].cumsum()
    total = s[-1]
    if total == 0:
        return 0
    return int(np.searchsorted(s, frac * total) + 1)


def prune_loss(counts: np.ndarray, keep: int) -> float:
    """Fraction of routed mass hitting the pruned (coldest) experts."""
    s = np.sort(counts)[::-1]
    total = s.sum()
    if total == 0:
        return 0.0
    return float(s[keep:].sum() / total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist-prefix", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.hist_prefix}.rank*.json"))
    if not paths:
        raise SystemExit(f"no histograms at {args.hist_prefix}.rank*.json")
    ranks = [json.load(open(p)) for p in paths]
    base = ranks[0]
    for other, path in zip(ranks[1:], paths[1:]):
        if other["layers"] != base["layers"]:
            print(f"WARNING: {path} disagrees with {paths[0]} (TP replicas "
                  "should route identically)")

    num_experts = base["num_experts"]
    rows = []
    agg = np.zeros(num_experts, dtype=np.int64)
    for layer_str, counts_list in sorted(base["layers"].items(), key=lambda i: int(i[0])):
        layer = int(layer_str)
        counts = np.asarray(counts_list, dtype=np.int64)
        tag = "hash" if layer in HASH_LAYERS else "gate"
        rows.append((layer, tag, counts))
        if layer not in HASH_LAYERS:
            agg += counts

    print(f"{'layer':>5} {'kind':>4} {'total':>10} {'c50':>4} {'c90':>4} "
          f"{'c99':>4} {'loss@-25%':>9} {'loss@-50%':>9}")
    for layer, tag, counts in rows:
        print(f"{layer:>5} {tag:>4} {int(counts.sum()):>10} "
              f"{coverage(counts, 0.5):>4} {coverage(counts, 0.9):>4} "
              f"{coverage(counts, 0.99):>4} "
              f"{prune_loss(counts, int(num_experts * 0.75)):>9.4f} "
              f"{prune_loss(counts, int(num_experts * 0.5)):>9.4f}")

    print("\nAggregate over gate-routed layers:")
    print(f"  experts for 50% of mass: {coverage(agg, 0.5)}/{num_experts}")
    print(f"  experts for 90% of mass: {coverage(agg, 0.9)}/{num_experts}")
    print(f"  experts for 99% of mass: {coverage(agg, 0.99)}/{num_experts}")
    print(f"  mass lost pruning coldest 25%: {prune_loss(agg, 192):.4%}")
    print(f"  mass lost pruning coldest 50%: {prune_loss(agg, 128):.4%}")
    gini = float(
        np.abs(np.subtract.outer(agg, agg)).sum() / (2 * len(agg) * agg.sum())
    ) if agg.sum() else 0.0
    print(f"  Gini coefficient: {gini:.3f} (0=uniform, 1=concentrated)")


if __name__ == "__main__":
    main()
