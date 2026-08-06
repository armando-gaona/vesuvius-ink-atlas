"""Paired comparison of inference recipes on identical segments.

Many segments were published more than once, under different recipes, from the same scan.
That is a controlled experiment the project already ran without framing it as one: scroll,
segment, surface and scan are held fixed, and only the recipe varies.

Comparing recipe means across the whole corpus would be confounded - the recipes are not
applied to the same segments in the same proportions. Pairing removes that entirely.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-csv", required=True)
    ap.add_argument("--score", default="mass_letter")
    ap.add_argument("--thresh", type=float, default=0.875)
    ap.add_argument("--min-windows", type=int, default=20)
    args = ap.parse_args()

    per = defaultdict(list)
    with open(args.atlas_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per[(r["scroll"], r["segment"], r["recipe"])].append(float(r[args.score]))

    stats = {k: (len(v), float(np.mean(v)), float(np.mean(np.array(v) >= args.thresh)))
             for k, v in per.items() if len(v) >= args.min_windows}

    by_seg = defaultdict(dict)
    for (scroll, seg, rec), v in stats.items():
        by_seg[(scroll, seg)][rec] = v

    pairs = {s: r for s, r in by_seg.items() if len(r) > 1}
    print(f"segments published under more than one recipe: {len(pairs)}")
    if not pairs:
        return

    combos = defaultdict(list)
    for (scroll, seg), recs in sorted(pairs.items()):
        names = sorted(recs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                combos[(a, b)].append((scroll, seg, recs[a], recs[b]))

    for (a, b), items in combos.items():
        da = np.array([x[2][2] for x in items])
        db = np.array([x[3][2] for x in items])
        print(f"\n=== {a}  vs  {b}   ({len(items)} paired segments) ===")
        print(f"  mean readable fraction   {a}: {da.mean():.3f}   {b}: {db.mean():.3f}")
        print(f"  {a} wins on {int((da > db).sum())}/{len(items)} segments")
        # Sign test: how surprising is that split if the recipes were equivalent? Reported
        # because a lopsided-looking count is easy to over-read, and because the first
        # version of this comparison produced a 36/36 sweep that turned out to be a unit bug
        # in the metric rather than a difference between recipes.
        n, k = len(items), int((da > db).sum())
        k = max(k, n - k)
        logp = [sum(np.log((n - j) / (j + 1)) for j in range(i)) - n * np.log(2)
                for i in range(k, n + 1)]
        p = min(1.0, 2 * float(np.exp(logp).sum()))
        print(f"  sign test  two-sided p = {p:.4g}   (n={n})")
        print(f"  median difference        {np.median(da - db):+.3f}")

        print(f"\n  {'segment':<30}{a[:11]:>12}{b[:11]:>12}{'diff':>9}")
        for scroll, seg, va, vb in sorted(items, key=lambda x: x[3][2] - x[2][2]):
            print(f"  {seg[:29]:<30}{va[2]:>12.3f}{vb[2]:>12.3f}{vb[2] - va[2]:>+9.3f}")


if __name__ == "__main__":
    main()
