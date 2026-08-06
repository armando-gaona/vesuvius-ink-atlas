"""Turn the atlas ranking into a classifier, using the blind labels.

Joins the hand labels to the scores they were produced without seeing, then answers the
only question that matters for publication: at what score does a window actually become
readable, and how wrong is that cutoff?

Reports AUC (threshold-free, so it cannot be tuned into looking good), then a full
threshold sweep with precision and recall, and finally what the chosen cutoff implies for
the corpus as a whole.
"""

import argparse
import csv

import numpy as np


def load(path, key=None):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    args = ap.parse_args()

    lab = {int(r["i"]): r["label"].strip() for r in load(args.labels_csv) if r["label"].strip()}
    key = load(args.key_csv)

    joined = []
    for r in key:
        i = int(r["i"])
        if i in lab:
            joined.append((i, r["scroll"], float(r[args.score]), lab[i]))
    if not joined:
        print("no labels yet")
        return

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

    atlas = load(args.atlas_csv)
    av = np.array([float(r[args.score]) for r in atlas])
    print(f"\ncorpus implication ({len(av)} windows):")
    for t in (0.75, 0.80, 0.85, 0.875, 0.90, 0.95):
        print(f"  >= {t:.3f}   {int((av >= t).sum()):>7}   {float((av >= t).mean()):>7.2%}")

    t = args.at
    by = {}
    for r in atlas:
        d = by.setdefault(r["scroll"], [0, 0])
        d[0] += 1
        d[1] += float(r[args.score]) >= t
    print(f"\nper scroll at the calibrated threshold {t:.3f}:")
    print(f"{'scroll':<14}{'windows':>9}{'text':>7}{'pct':>8}")
    for k, d in sorted(by.items(), key=lambda kv: -kv[1][1] / kv[1][0]):
        print(f"{k:<14}{d[0]:>9}{d[1]:>7}{d[1] / d[0]:>8.2%}")


if __name__ == "__main__":
    main()
