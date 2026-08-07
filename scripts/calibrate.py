"""Turn the atlas ranking into a classifier, using the blind labels.

Joins the hand labels to the scores they were produced without seeing, then answers the
only question that matters for publication: at what score does a window actually become
readable, and how wrong is that cutoff?

Reports AUC (threshold-free, so it cannot be tuned into looking good), then a full
threshold sweep with precision and recall, and finally what the chosen cutoff implies for
the corpus as a whole.

Both scores come from the atlas, never from the labeling key. The key stores a snapshot of
the score at sampling time; if the metric is fixed afterwards, that snapshot goes stale and
the calibration silently reports a number the published atlas cannot reproduce. That is
exactly what happened to six PHerc0172 windows and inflated the published AUC from 0.918
to 0.930 (see README, "Corrections").
"""

import argparse
import csv

import numpy as np


def load(path, key=None):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tiled(rows):
    """Keep a non-overlapping subset of the windows.

    The atlas strides by half a window in both axes, so every point of papyrus is covered
    about four times and adjacent rows are not independent. Any confidence interval over
    the full grid is therefore about twice as narrow as it should be. Dropping the odd
    half-steps leaves a plain tiling: same physical area per window, no shared pixels.
    """
    out = []
    for r in rows:
        step = int(r["window_px"]) // 2
        if (int(r["y0"]) // step) % 2 == 0 and (int(r["x0"]) // step) % 2 == 0:
            out.append(r)
    return out


def wilson(k, n, z=1.96):
    """Wilson interval: behaves at k=0, which the binomial normal approximation does not."""
    if not n:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def auc(pos, neg):
    """Mann-Whitney U: the probability a random positive outranks a random negative."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties, otherwise ties are scored arbitrarily
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-csv", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--atlas-csv", required=True)
    ap.add_argument("--score", default="mass_letter")
    ap.add_argument("--at", type=float, default=0.900,
                    help="Calibrated threshold to apply to the corpus")
    # mass_letter is a FRACTION of ink mass, so a window holding almost no ink can still
    # score high if the little it holds lands in the letter band. The floor refuses to score
    # those. It is a guard, not a measured improvement: on 125 labels the AUC gain is
    # +0.013 with a bootstrap CI of [0.000, 0.041] and comes down to a single window.
    ap.add_argument("--ink-floor", type=float, default=0.05,
                    help="Score 0 below this frac_above. 0 disables.")
    ap.add_argument("--all-windows", action="store_true",
                    help="Use the full overlapping grid instead of a non-overlapping tiling")
    args = ap.parse_args()

    lab = {int(r["i"]): r["label"].strip() for r in load(args.labels_csv) if r["label"].strip()}
    key = load(args.key_csv)

    atlas = load(args.atlas_csv)
    at = {(r["scroll"], r["segment"], r["recipe"], r["y0"], r["x0"]): r for r in atlas}

    def score(r):
        if args.ink_floor and float(r["frac_above"]) < args.ink_floor:
            return 0.0
        return float(r[args.score])

    joined, stale = [], 0
    for r in key:
        i = int(r["i"])
        if i not in lab:
            continue
        a = at[(r["scroll"], r["segment"], r["recipe"], r["y0"], r["x0"])]
        stale += abs(float(a[args.score]) - float(r[args.score])) > 1e-6
        joined.append((i, r["scroll"], score(a), lab[i]))
    if not joined:
        print("no labels yet")
        return
    if stale:
        print(f"WARNING: {stale} labelled windows carry a stale score in the key; "
              f"the atlas value is used")

    s = np.array([j[2] for j in joined])
    y = [j[3] for j in joined]
    counts = {k: y.count(k) for k in ("text", "partial", "speckle")}
    print(f"labelled: {len(joined)}   " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    print(f"\n{'label':<10}{'n':>5}{'mean':>8}{'p25':>8}{'median':>8}{'p75':>8}")
    for k in ("text", "partial", "speckle"):
        v = s[[i for i, t in enumerate(y) if t == k]]
        if len(v):
            print(f"{k:<10}{len(v):>5}{v.mean():>8.3f}{np.percentile(v, 25):>8.3f}"
                  f"{np.median(v):>8.3f}{np.percentile(v, 75):>8.3f}")

    # Two readings of "legible". The strict one is the honest headline; the loose one shows
    # how much the answer depends on where the line is drawn.
    for name, pos_set in (("text vs rest", {"text"}),
                          ("text+partial vs speckle", {"text", "partial"})):
        pos = s[[i for i, t in enumerate(y) if t in pos_set]]
        neg = s[[i for i, t in enumerate(y) if t not in pos_set]]
        print(f"\n=== {name} ===")
        print(f"AUC = {auc(pos, neg):.3f}   (0.5 = worthless, 1.0 = perfect)")
        print(f"{'thresh':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>7}{'recall':>8}{'F1':>7}")
        best = None
        for t in np.arange(0.30, 1.00, 0.025):
            tp = int((pos >= t).sum())
            fp = int((neg >= t).sum())
            fn = int((pos < t).sum())
            if tp == 0:
                continue
            p, rc = tp / (tp + fp), tp / (tp + fn)
            f1 = 2 * p * rc / (p + rc)
            if best is None or f1 > best[0]:
                best = (f1, t, p, rc)
            print(f"{t:>7.3f}{tp:>5}{fp:>5}{fn:>5}{p:>7.3f}{rc:>8.3f}{f1:>7.3f}")
        if best:
            print(f"best F1 = {best[0]:.3f} at thresh {best[1]:.3f} "
                  f"(precision {best[2]:.3f}, recall {best[3]:.3f})")

    corpus = atlas if args.all_windows else tiled(atlas)
    av = np.array([score(r) for r in corpus])
    print(f"\ncorpus implication ({len(av)} windows"
          f"{'' if args.all_windows else f', tiled from {len(atlas)}'}):")
    for t in (0.75, 0.80, 0.85, 0.875, 0.90, 0.95):
        k = int((av >= t).sum())
        lo, hi = wilson(k, len(av))
        print(f"  >= {t:.3f}   {k:>7}   {k / len(av):>7.2%}   CI95 [{lo:.2%}, {hi:.2%}]")

    t = args.at
    by = {}
    for r, v in zip(corpus, av):
        d = by.setdefault(r["scroll"], [0, 0])
        d[0] += 1
        d[1] += v >= t
    print(f"\nper scroll at the calibrated threshold {t:.3f}:")
    print(f"{'scroll':<14}{'windows':>9}{'text':>7}{'pct':>8}{'CI95':>18}")
    for k, d in sorted(by.items(), key=lambda kv: -kv[1][1] / kv[1][0]):
        lo, hi = wilson(d[1], d[0])
        print(f"{k:<14}{d[0]:>9}{d[1]:>7}{d[1] / d[0]:>8.2%}"
              f"{f'[{lo:.2%}, {hi:.2%}]':>18}")

    # A prediction is the unit anyone can act on, and `key` is what identifies one. Grouping
    # by (scroll, segment, recipe) merges distinct models rendered under the same label.
    per = {}
    for r, v in zip(corpus, av):
        d = per.setdefault(r["key"], [0, 0])
        d[0] += 1
        d[1] += v >= t
    elig = {k: v for k, v in per.items() if v[0] >= 20}
    zero = [k for k, v in elig.items() if v[1] == 0]
    print(f"\npredictions with >= 20 tiled windows: {len(elig)}   "
          f"with no readable window at all: {len(zero)} ({len(zero) / len(elig):.1%})")


if __name__ == "__main__":
    main()
