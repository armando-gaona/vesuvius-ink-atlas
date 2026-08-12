"""How much of the AUC gain is real, once neighbouring pixels stop being counted as independent.

    python scripts/bootstrap_auc.py <labels_dir> <segment> \
        --base seed42.tif,seed43.tif --soup soup42.tif,soup43.tif \
        [--block 256] [--resamples 1000] [--seed 0]

The held-out region has a few hundred thousand pixels, which looks like an enormous sample. It
is not. Ink is written in strokes: if a pixel is inked, its neighbour almost certainly is too,
and the model's error is correlated over the same distance. Treating those pixels as independent
makes any AUC look far more precise than it is, and the gain measured here (+0.010 to +0.019) is
small enough that the difference matters.

So the resampling unit is a square TILE of the image, not a pixel. Tiles are drawn with
replacement, the AUC is recomputed on each resampled set, and the whole thing is repeated. The
spread of the resulting deltas is an honest interval, because whatever correlation lives inside
a tile is carried along instead of being broken up.

The statistic bootstrapped is the same one that is reported: the mean of the two seeds' AUC on
the soup, minus the mean of the two seeds' AUC on the baseline, on the held-out region only.
Both seeds are resampled on the SAME tiles in each replicate, because they are two measurements
of one papyrus and not two independent experiments.

Block size is a judgement call and there is no right answer, so two are reported. A block too
small still splits correlated pixels and the interval comes out too narrow; a block too large
leaves so few blocks to draw from that the bootstrap itself stops meaning anything. Both failure
modes are visible in the printed number of blocks: single digits are a warning, not a result.
"""
import argparse
import os

import numpy as np
import tifffile
import zarr
from scipy.stats import rankdata

PLANE_DEFAULT = 10


def load_pred(path, shape):
    q = tifffile.imread(path)
    if q.dtype == np.uint8:
        q = q.astype(np.float32) / 255.0
    elif q.dtype == np.uint16:
        q = q.astype(np.float32) / 65535.0
    else:
        q = q.astype(np.float32)
    if q.shape != shape:
        raise SystemExit(f"shape mismatch for {path}: {q.shape} vs {shape}")
    return q


def auc(scores, labels):
    npos = int(labels.sum())
    nneg = labels.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir")
    ap.add_argument("segment")
    ap.add_argument("--base", required=True, help="baseline TIFFs, comma separated, one per seed")
    ap.add_argument("--soup", required=True, help="soup TIFFs, comma separated, same order")
    ap.add_argument("--block", type=int, nargs="+", default=[256, 512], help="block side in px")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plane", type=int, default=PLANE_DEFAULT)
    a = ap.parse_args()

    def open_(name):
        return zarr.open(os.path.join(a.labels_dir, f"{a.segment}_{name}.zarr", "0"), mode="r")

    ink = open_("inklabels")
    shape = tuple(ink.shape[1:])
    y = np.asarray(ink[a.plane]) > 0
    m = np.asarray(open_("supervision_mask")[a.plane]) > 0
    v = np.asarray(open_("validation_mask")[a.plane]) > 0
    held = v & ~m

    base_paths = [p for p in a.base.split(",") if p]
    soup_paths = [p for p in a.soup.split(",") if p]
    if len(base_paths) != len(soup_paths):
        raise SystemExit("one baseline per soup is needed, in the same order")

    base = [load_pred(p, shape) for p in base_paths]
    soup = [load_pred(p, shape) for p in soup_paths]

    rows, cols = np.nonzero(held)
    yy = y[held]
    bb = [p[held] for p in base]
    ss = [p[held] for p in soup]

    obs = np.mean([auc(s, yy) for s in ss]) - np.mean([auc(b, yy) for b in bb])
    print(f"{a.segment}   held-out pixels {yy.size:,}   ink {100.0 * yy.mean():.2f}%")
    print(f"observed delta (mean of {len(ss)} seeds) {obs:+.6f}")

    for block in a.block:
        key = (rows // block).astype(np.int64) * 100000 + (cols // block)
        uniq, inv = np.unique(key, return_inverse=True)
        # index list per block, so a replicate is a concatenation and not a mask scan
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv, minlength=uniq.size)
        starts = np.concatenate([[0], np.cumsum(counts)])
        members = [order[starts[i]:starts[i + 1]] for i in range(uniq.size)]

        rng = np.random.default_rng(a.seed)
        deltas = np.empty(a.resamples)
        for it in range(a.resamples):
            pick = rng.integers(0, uniq.size, uniq.size)
            idx = np.concatenate([members[j] for j in pick])
            yb = yy[idx]
            if yb.all() or not yb.any():
                deltas[it] = np.nan
                continue
            deltas[it] = (np.mean([auc(s[idx], yb) for s in ss])
                          - np.mean([auc(b[idx], yb) for b in bb]))
        d = deltas[~np.isnan(deltas)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  block {block:4d} px   n_blocks {uniq.size:4d}   "
              f"CI95 [{lo:+.4f}, {hi:+.4f}]   P(delta>0) {float((d > 0).mean()):.3f}   "
              f"valid {d.size}/{a.resamples}")
        if uniq.size < 20:
            print(f"  {'':13s} WARNING: {uniq.size} blocks is too few to resample from. "
                  f"Read this interval as an order of magnitude, not as a bound.")


if __name__ == "__main__":
    main()
