"""Why does a scroll score the way it does - is it the prediction, or the metric?

The atlas reports PHerc0172 at 0% readable. That scroll has already produced two false
findings from unit errors, and its contact sheets show clean Greek, so "0%" needs a cause
before it is published as a result. The score is a mass fraction inside a size band, so a
scroll can score low in exactly three ways: mass below the band, mass above it (merged into
smears), or almost no mass at all. Those are different failures and only one of them is the
pipeline's.

Splitting the score into its complements per scroll says which it is, with no new labels.
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-csv", required=True)
    ap.add_argument("--score", default="mass_letter")
    args = ap.parse_args()

    per = defaultdict(list)
    with open(args.atlas_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per[r["scroll"]].append(r)

    cols = ["mass_letter", "mass_smear", "mass_big", "frac_above", "median_extent",
            "n_comp", "thresh"]
    print(f"{'scroll':<14}{'n':>6}{'pitch':>7}" + "".join(f"{c[:9]:>11}" for c in cols)
          + f"{'below band':>12}")
    for scroll, rs in sorted(per.items(), key=lambda kv: -len(kv[1])):
        v = {c: np.array([float(r[c]) for r in rs]) for c in cols}
        pitch = np.median([float(r["pitch_um"]) for r in rs])
        # Mass that is neither a letter nor a smear is mass in components too SMALL to be
        # a letter - speckle. The three fractions sum to 1 by construction.
        below = 1.0 - v["mass_letter"] - v["mass_smear"]
        print(f"{scroll:<14}{len(rs):>6}{pitch:>7.2f}"
              + "".join(f"{v[c].mean():>11.3f}" for c in cols)
              + f"{below.mean():>12.3f}")

    print("\nletter band in ds8 pixels, per pitch (band is 960-7200 um):")
    seen = set()
    for scroll, rs in per.items():
        p = float(np.median([float(r["pitch_um"]) for r in rs]))
        if p in seen:
            continue
        seen.add(p)
        print(f"  pitch {p:>6.3f} um -> a letter (~3000 um) is "
              f"{3000 / (p * 8):>6.1f} ds8 px;  band = "
              f"{960 / (p * 8):>5.1f} .. {7200 / (p * 8):>6.1f} px")


if __name__ == "__main__":
    main()
