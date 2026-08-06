"""Re-attach the blind labels to a rebuilt atlas.

The labels describe pixels, so they survive any change to the scoring code. The score stored
in the labeling key does not - it is a snapshot of the atlas at sampling time. Re-joining by
(scroll, segment, recipe, y0, x0) keeps the calibration honest after a metric fix, without
re-labelling anything and without the labeller ever seeing the new numbers.

Prints how far each window moved, because a fix that changes nothing was not a fix and a fix
that changes everything deserves a second look before its output is believed.
"""

import argparse
import csv

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-csv", required=True)
    ap.add_argument("--atlas-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--score", default="mass_letter")
    args = ap.parse_args()

    atlas = {}
    with open(args.atlas_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            atlas[(r["scroll"], r["segment"], r["recipe"], r["y0"], r["x0"])] = r

    with open(args.key_csv, encoding="utf-8") as f:
        key = list(csv.DictReader(f))

    out, miss, moved = [], 0, []
    for r in key:
        k = (r["scroll"], r["segment"], r["recipe"], r["y0"], r["x0"])
        a = atlas.get(k)
        if a is None:
            miss += 1
            continue
        old = float(r[args.score])
        new = float(a[args.score])
        moved.append(new - old)
        out.append({**r, "old_" + args.score: old, args.score: new,
                    "pitch_um": a.get("pitch_um", ""), "recipe": r["recipe"]})

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"wrote {args.out_csv}   rejoined {len(out)}   unmatched {miss}")

    d = np.array(moved)
    print(f"score change: mean {d.mean():+.4f}  median {np.median(d):+.4f}  "
          f"max |d| {np.abs(d).max():.4f}   changed at all: {int((np.abs(d) > 1e-4).sum())}")

    by_rec = {}
    for r, dd in zip(out, d):
        by_rec.setdefault(r["recipe"], []).append(dd)
    print(f"\n{'recipe':<12}{'n':>5}{'mean delta':>12}")
    for k, v in sorted(by_rec.items()):
        print(f"{k:<12}{len(v):>5}{np.mean(v):>+12.4f}")


if __name__ == "__main__":
    main()
