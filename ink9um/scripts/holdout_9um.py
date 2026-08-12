"""Separate what the model memorised from what it actually generalises, inside one segment.

    python scripts/holdout_9um.py <prediction.tif> <labels_dir> <segment> [--plane 10]

Why: on the five segments with native annotation the AUC comes out above 0.995, but those
segments are the published annotated set itself, so the model almost certainly trained on them
and that number says nothing about new data.

Three segments of the aligned set also ship `_validation_mask.zarr`, which marks the held-out
region. Per the ink_9um README these are "the online-validation cases the released checkpoints
report metrics on". That allows a clean comparison, because the two regions sit on the SAME
papyrus, the SAME scan, the SAME z window and the SAME annotation. The only thing that differs
between them is whether the model saw them.

The AUC drop between the two regions is the honest measure of how much of that 0.99 was memory.

The aligned set has 21 planes and is annotated only at Z=10 (the native set has 28 and is
annotated at Z=14), so the default plane here is different on purpose.
"""
import argparse
import os

import numpy as np
import tifffile
import zarr
from scipy.stats import rankdata

PITCH_UM = 9.596


def to_unit(q):
    """Bring a prediction to [0, 1] using its dtype, which is metadata, not its content."""
    if q.dtype == np.uint8:
        return q.astype(np.float32) / 255.0
    if q.dtype == np.uint16:
        return q.astype(np.float32) / 65535.0
    if q.dtype in (np.float32, np.float64):
        return q.astype(np.float32)
    raise SystemExit(f"unsupported dtype {q.dtype}: the scale would have to be guessed")


def auc(scores, labels):
    r = rankdata(scores)
    npos = int(labels.sum())
    nneg = labels.size - npos
    if npos == 0 or nneg == 0:
        return float("nan"), npos, nneg
    return (r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg), npos, nneg


def block(name, s, t):
    a_, npos, nneg = auc(s, t)
    if npos == 0 or nneg == 0:
        print(f"{name:28s} n={s.size:>10,}   only one class present, no AUC")
        return
    print(f"{name:28s} n={s.size:>10,}   AUC {a_:.4f}   ink {100.0 * t.mean():5.2f}%   "
          f"mean on ink {s[t].mean():.4f}  off ink {s[~t].mean():.4f}")

    def f1_at(th):
        pr = s > th
        tp = int((pr & t).sum())
        if tp == 0:
            return 0.0
        prec = tp / int(pr.sum())
        rec = tp / npos
        return 2 * prec * rec / (prec + rec)

    best_f1, best_th = 0.0, float("nan")
    for th in np.arange(0.30, 0.75, 0.05):
        f1 = f1_at(th)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    # the default 0.50 is printed separately: the cost of using it on new data is the finding
    print(f"{'':28s} best F1 {best_f1:.3f} at threshold {best_th:.2f}   "
          f"F1 at the default 0.50 {f1_at(0.50):.3f}")

    # balanced accuracy = (sensitivity + specificity) / 2, at threshold 0.50.
    # This is the metric the checkpoints' own config declares in `best_checkpoint_metric`, so it
    # is reported alongside the AUC rather than instead of it.
    pr = s > 0.50
    sens = float((pr & t).sum()) / npos
    spec = float((~pr & ~t).sum()) / nneg
    print(f"{'':28s} balanced accuracy at 0.50 {0.5 * (sens + spec):.4f}   "
          f"(sensitivity {sens:.4f}  specificity {spec:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="one or more TIFFs separated by commas; several are averaged")
    ap.add_argument("labels_dir")
    ap.add_argument("segment")
    ap.add_argument("--plane", type=int, default=10)
    a = ap.parse_args()

    paths = [r for r in a.pred.split(",") if r]
    total, dtypes = None, []
    for r in paths:
        q = tifffile.imread(r)
        dtypes.append(q.dtype)
        # Normalise by DTYPE, never by the observed maximum. A float prediction whose maximum
        # happens to be 1.6 is not an 8-bit image, and a uint8 prediction that never exceeds 1
        # (an almost blank one, which is exactly the interesting case) would be left unscaled.
        # Deciding the scale from the data makes the scale depend on the result.
        q = to_unit(q)
        total = q if total is None else total + q
    if len(set(dtypes)) > 1:
        raise SystemExit(
            f"mixed dtypes across the inputs ({', '.join(str(d) for d in dtypes)}); "
            "averaging them would silently mix two different quantisations")
    p = total / len(paths)
    if len(paths) > 1:
        print(f"average of {len(paths)} predictions")

    def open_(name):
        return zarr.open(os.path.join(a.labels_dir, f"{a.segment}_{name}.zarr", "0"), mode="r")

    ink = open_("inklabels")
    sup = open_("supervision_mask")
    val = open_("validation_mask")
    print(f"prediction {p.shape}   labels {ink.shape}")
    if p.shape != tuple(ink.shape[1:]):
        raise SystemExit("shapes do not match: comparing them would be an alignment bug")

    y = np.asarray(ink[a.plane]) > 0
    m = np.asarray(sup[a.plane]) > 0
    v = np.asarray(val[a.plane]) > 0

    # The two masks are disjoint: `supervision_mask` marks the looked-at region that goes into
    # training and `validation_mask` marks the looked-at region that is held out. Neither is a
    # subset of the other, so the held-out region is taken as is.
    train = m & ~v
    held = v & ~m
    overlap = m & v
    print(f"plane Z={a.plane}   supervised {100.0 * m.mean():.2f}%   held out {100.0 * v.mean():.2f}%   "
          f"overlap between the two {100.0 * overlap.mean():.3f}%")
    print()
    block("supervised (seen)", p[train], y[train])
    block("HELD OUT (unseen)", p[held], y[held])


if __name__ == "__main__":
    main()
