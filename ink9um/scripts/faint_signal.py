"""Does the weight soup buy clearness by losing faint ink?

    python scripts/faint_signal.py <labels_dir> <out_dir> [--segment NAME] [--resamples 1000]

Pre-registered in `../protocols/faint_signal_protocol.md`, written before this script existed and
before any of these quantities was computed.

The objection being tested was raised by a Vesuvius team member: an ensemble can gain "clearness"
by smoothing the letters while losing fainter signal. It bites because the published AUC is taken
over every pixel of the held-out region, so it is dominated by thick obvious ink. The soup could
raise it while degrading exactly the weak ink that matters for finding new text.

Two things are measured and kept apart on purpose:

  Q1  recall on hard ink, at a MATCHED false positive rate
  Q2  whether the soup output is literally smoother

A prediction can be smoother without being blinder, and that combination is the interesting answer.

Three design points carry the whole thing:

  * Difficulty is defined by the OTHER seed's baseline, never by the model under test. Ranking ink
    by the confidence of the model being judged would guarantee the answer.
  * Recall is compared at a matched FPR, never at a shared numeric threshold. A prediction whose
    values are merely shifted upward would "recover more faint ink" for free.
  * Everything is repeated on non-ink pixels. If the soup lifts those in the same pattern it is
    firing more everywhere rather than recovering weak signal.
"""
import argparse
import os
import sys

import numpy as np
import tifffile
import zarr
from scipy.stats import rankdata
from scipy import ndimage

PITCH_UM = 9.596
PLANE = 10
SEEDS = (42, 43)
FPR_LEVELS = (0.01, 0.05, 0.10)
FPR_PRIMARY = 0.05
N_STRATA = 4
FINE_SCALE_UM = 192.0          # secondary spectral cut, finer than a stroke
GATE_TOL = 5e-4                # published values are quoted to 4 decimals

SEGMENTS = ("pherc0814-46527", "pherc0139-w016", "pherc1667-w029")

# Published held-out AUCs, per seed. Baselines from holdout9_results.md, A4 from soup_results.md.
# The gate exists because several soup builds sit on disk (float32 and float64 accumulations of the
# same soup) and silently picking the wrong file would answer a different question.
PUBLISHED = {
    "pherc0814-46527": {"base": (0.8683, 0.8433), "soup": (0.8692, 0.8630)},
    "pherc0139-w016":  {"base": (0.8014, 0.9087), "soup": (0.8526, 0.8960)},
    "pherc1667-w029":  {"base": (0.8717, 0.9197), "soup": (0.8966, 0.9297)},
}


def to_unit(q):
    """Scale by DTYPE, never by the observed maximum: deciding the scale from the data would make
    the scale depend on the result."""
    if q.dtype == np.uint8:
        return q.astype(np.float32) / 255.0
    if q.dtype == np.uint16:
        return q.astype(np.float32) / 65535.0
    if q.dtype in (np.float32, np.float64):
        return q.astype(np.float32)
    raise SystemExit(f"unsupported dtype {q.dtype}")


def find_pred(out_dir, segment, seed, kind):
    """Resolve a prediction path. Naming on disk is not uniform for pherc0814-46527, and the soup
    has both float32 and float64 builds; the float64 one is what produced the published tables."""
    short = "0814" if segment.startswith("pherc0814") else segment
    if kind == "base":
        cands = [f"{segment}_seed{seed}_s75k.tif"]
    else:
        cands = [f"soup_{segment}_A4_seed{seed}_f64.tif",
                 f"soup_{short}_A4_seed{seed}_f64.tif",
                 f"soup_{segment}_A4_seed{seed}.tif",
                 f"soup_{short}_A4_seed{seed}.tif"]
    for c in cands:
        p = os.path.join(out_dir, c)
        if os.path.exists(p):
            return p
    raise SystemExit(f"no {kind} prediction found for {segment} seed{seed}; tried {cands}")


def auc(scores, labels):
    npos = int(labels.sum())
    nneg = labels.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def recall_at_matched_fpr(pred_ink, pred_non, strata_ink, fpr):
    """Threshold set so the FPR on non-ink equals `fpr`, then recall inside each stratum.

    Returns (recalls per stratum, overall recall, achieved fpr). The achieved FPR is returned and
    printed because these predictions are quantised, so ties can keep it from landing exactly on
    the target, and hiding that would misrepresent how matched the comparison really is.
    """
    thr = float(np.quantile(pred_non, 1.0 - fpr))
    achieved = float((pred_non > thr).mean())
    hit = pred_ink > thr
    out = np.full(N_STRATA, np.nan)
    for s in range(N_STRATA):
        m = strata_ink == s
        if m.any():
            out[s] = float(hit[m].mean())
    return out, float(hit.mean()), achieved


def quartiles(values, n=N_STRATA):
    """Rank-based split, so it is robust to quantisation and to the shape of the distribution.
    Stratum 0 is the lowest score."""
    r = rankdata(values, method="average")
    return np.minimum((r / (r.size + 1) * n).astype(np.int64), n - 1)


def gradient_energy(pred2d, mask2d):
    """Mean squared gradient magnitude over the interior of the mask.

    The prediction is standardised over the held-out region first, so a global change of scale or
    offset cannot register as a change of sharpness. The mask is eroded by one pixel so that the
    step at the mask boundary, which is an artifact of where we cut and not a property of the
    prediction, does not enter the average.
    """
    vals = pred2d[mask2d]
    z = (pred2d - vals.mean()) / (vals.std() + 1e-12)
    gy, gx = np.gradient(z)
    interior = ndimage.binary_erosion(mask2d, np.ones((3, 3), bool))
    if not interior.any():
        return float("nan"), 0
    e = (gy[interior] ** 2 + gx[interior] ** 2)
    return float(e.mean()), int(interior.sum())


def fine_fraction(pred2d, mask2d):
    """Fraction of spectral energy above FINE_SCALE_UM, on the largest connected piece.

    A bounding box is NOT the mask: the held-out region can arrive as pieces in opposite corners,
    where the box is mostly empty and its spectrum would be the spectrum of the emptiness. So the
    largest connected component is taken, its box fill is measured, and the whole secondary is
    dropped rather than reported if the box is not essentially full.
    """
    lab, n = ndimage.label(mask2d)
    if n == 0:
        return None, "no connected component"
    sizes = ndimage.sum(mask2d, lab, range(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == big)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    box = (lab[y0:y1, x0:x1] == big)
    fill = float(box.mean())
    h, w = box.shape
    if fill < 0.80:
        return None, f"largest component fills only {100 * fill:.1f}% of its box"
    if min(h, w) < 64:
        return None, f"largest component is {h}x{w} px, too small for a stable spectrum"
    patch = pred2d[y0:y1, x0:x1].astype(np.float64)
    patch = (patch - patch.mean()) / (patch.std() + 1e-12)
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(patch * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)                     # cycles per pixel
    f_cut = PITCH_UM / FINE_SCALE_UM                   # cycles per pixel at the cut scale
    tot = P.sum()
    return float(P[f > f_cut].sum() / tot), f"{h}x{w} px, box fill {100 * fill:.1f}%"


def bootstrap_primary(rows, cols, ink, strata, preds, block, resamples, rng_seed):
    """Block bootstrap of the primary statistic: delta of hardest-quartile recall at matched FPR,
    meaned over the two seeds. Tiles are resampled, not pixels, because ink comes in strokes and
    neighbouring errors are not independent. Both seeds ride the same tiles in every replicate,
    since they are two measurements of one papyrus.

    The threshold is recomputed inside each replicate: it is part of the statistic, so freezing it
    would understate the uncertainty.
    """
    key = (rows // block).astype(np.int64) * 100003 + (cols // block)
    uniq, inv = np.unique(key, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=uniq.size)
    starts = np.concatenate([[0], np.cumsum(counts)])
    members = [order[starts[i]:starts[i + 1]] for i in range(uniq.size)]

    rng = np.random.default_rng(rng_seed)
    out = np.empty(resamples)
    for it in range(resamples):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([members[j] for j in pick])
        yb = ink[idx]
        if yb.all() or not yb.any():
            out[it] = np.nan
            continue
        d = []
        for seed in SEEDS:
            st = strata[seed][idx][yb]
            hard = st == 0
            if not hard.any():
                d = []
                break
            vals = []
            for kind in ("soup", "base"):
                p = preds[(kind, seed)][idx]
                thr = np.quantile(p[~yb], 1.0 - FPR_PRIMARY)
                vals.append(float((p[yb][hard] > thr).mean()))
            d.append(vals[0] - vals[1])
        out[it] = np.mean(d) if d else np.nan
    d = out[~np.isnan(out)]
    if d.size == 0:
        return None
    lo, hi = np.percentile(d, [2.5, 97.5])
    return uniq.size, float(lo), float(hi), float((d > 0).mean()), d.size


def run_segment(labels_dir, out_dir, segment, resamples, rng_seed):
    print("=" * 100)
    print(f"SEGMENT {segment}")
    print("=" * 100)

    def open_(name):
        # The other scripts are invoked with labels_dir already pointing at the segment folder.
        # Both layouts are accepted here so the call site cannot silently pick a different segment.
        for root in (os.path.join(labels_dir, segment), labels_dir):
            p = os.path.join(root, f"{segment}_{name}.zarr", "0")
            if os.path.exists(p):
                return zarr.open(p, mode="r")
        raise SystemExit(f"cannot find {segment}_{name}.zarr under {labels_dir}")

    ink_z = open_("inklabels")
    shape = tuple(ink_z.shape[1:])
    y2d = np.asarray(ink_z[PLANE]) > 0
    m2d = np.asarray(open_("supervision_mask")[PLANE]) > 0
    v2d = np.asarray(open_("validation_mask")[PLANE]) > 0
    held2d = v2d & ~m2d

    preds2d, dtypes = {}, set()
    for kind in ("base", "soup"):
        for seed in SEEDS:
            p = find_pred(out_dir, segment, seed, kind)
            q = tifffile.imread(p)
            dtypes.add(str(q.dtype))
            q = to_unit(q)
            if q.shape != shape:
                raise SystemExit(f"shape mismatch {p}: {q.shape} vs {shape}")
            preds2d[(kind, seed)] = q
            print(f"  {kind:5s} seed{seed}  {os.path.basename(p)}")
    print(f"  dtypes {sorted(dtypes)}   held-out {100.0 * held2d.mean():.2f}% of the plane")

    rows, cols = np.nonzero(held2d)
    ink = y2d[held2d]
    preds = {k: v[held2d] for k, v in preds2d.items()}
    print(f"  held-out pixels {ink.size:,}   ink {100.0 * ink.mean():.2f}%")

    # ---------------------------------------------------------------- gate
    print("\n  GATE: reproduce the published held-out AUCs")
    ok = True
    for kind in ("base", "soup"):
        for i, seed in enumerate(SEEDS):
            got = auc(preds[(kind, seed)], ink)
            exp = PUBLISHED[segment][kind][i]
            good = abs(got - exp) <= GATE_TOL
            ok &= good
            print(f"    {kind:5s} seed{seed}  got {got:.4f}  published {exp:.4f}  "
                  f"{'ok' if good else 'MISMATCH'}")
    if not ok:
        raise SystemExit("gate failed: these are not the files that produced the published table")

    # ------------------------------------------------- Q1, cross-seed strata
    # Difficulty from the OTHER seed's baseline. Stratum 0 is the hardest ink.
    strata_ink, strata_non = {}, {}
    for seed in SEEDS:
        other = SEEDS[1] - (seed - SEEDS[0])
        ref = preds[("base", other)]
        s = np.full(ink.size, -1, np.int64)
        s[ink] = quartiles(ref[ink])
        s[~ink] = quartiles(ref[~ink])
        strata_ink[seed] = s
        strata_non[seed] = s

    print("\n  Q1  recall on ink by difficulty, at MATCHED false positive rate")
    print("      strata from the OTHER seed's baseline; stratum 0 = hardest ink")
    for fpr in FPR_LEVELS:
        tag = "PRIMARY" if fpr == FPR_PRIMARY else "       "
        print(f"\n    FPR {100 * fpr:.0f}%  {tag}")
        print(f"      {'seed':6s} {'cfg':5s} " + " ".join(f"{'Q' + str(i):>8s}" for i in range(N_STRATA))
              + f" {'all':>8s} {'achFPR':>8s}")
        deltas_hard = []
        for seed in SEEDS:
            r = {}
            for kind in ("base", "soup"):
                p = preds[(kind, seed)]
                rec, allrec, ach = recall_at_matched_fpr(p[ink], p[~ink], strata_ink[seed][ink], fpr)
                r[kind] = rec
                print(f"      {seed:<6d} {kind:5s} " + " ".join(f"{x:8.4f}" for x in rec)
                      + f" {allrec:8.4f} {100 * ach:7.3f}%")
            d = r["soup"] - r["base"]
            deltas_hard.append(d[0])
            print(f"      {'':6s} {'DELTA':5s} " + " ".join(f"{x:+8.4f}" for x in d))
        print(f"      hardest-quartile delta, mean of seeds: {np.mean(deltas_hard):+.4f}")

    # --------------------------------------------- Q1, geometric cross-check
    edt = ndimage.distance_transform_edt(y2d)[held2d]
    geo = np.full(ink.size, -1, np.int64)
    geo[ink] = quartiles(edt[ink])
    print("\n    geometric cross-check: strata by distance to nearest non-ink pixel")
    print(f"      stratum 0 = thinnest/edge ink.  FPR {100 * FPR_PRIMARY:.0f}%")
    for seed in SEEDS:
        r = {}
        for kind in ("base", "soup"):
            p = preds[(kind, seed)]
            rec, _, _ = recall_at_matched_fpr(p[ink], p[~ink], geo[ink], FPR_PRIMARY)
            r[kind] = rec
            print(f"      {seed:<6d} {kind:5s} " + " ".join(f"{x:8.4f}" for x in rec))
        print(f"      {'':6s} {'DELTA':5s} " + " ".join(f"{x:+8.4f}" for x in (r['soup'] - r['base'])))

    # ------------------------------------------------------ negative control
    print("\n  NEGATIVE CONTROL: false positive rate on NON-ink, by the same strata")
    print(f"      stratum 3 = most confusable non-ink.  FPR {100 * FPR_PRIMARY:.0f}%")
    for seed in SEEDS:
        r = {}
        for kind in ("base", "soup"):
            p = preds[(kind, seed)]
            thr = float(np.quantile(p[~ink], 1.0 - FPR_PRIMARY))
            fp = p[~ink] > thr
            st = strata_non[seed][~ink]
            rec = np.array([float(fp[st == s].mean()) if (st == s).any() else np.nan
                            for s in range(N_STRATA)])
            r[kind] = rec
            print(f"      {seed:<6d} {kind:5s} " + " ".join(f"{x:8.4f}" for x in rec))
        print(f"      {'':6s} {'DELTA':5s} " + " ".join(f"{x:+8.4f}" for x in (r['soup'] - r['base'])))

    # ------------------------------------------------------------------- Q2
    print("\n  Q2  is the soup output measurably smoother?")
    for seed in SEEDS:
        line = []
        for kind in ("base", "soup"):
            g, npix = gradient_energy(preds2d[(kind, seed)], held2d)
            line.append(g)
        print(f"      seed{seed}  gradient energy  base {line[0]:.5f}   soup {line[1]:.5f}   "
              f"ratio soup/base {line[1] / line[0]:.4f}")
    for seed in SEEDS:
        vals, notes = [], []
        for kind in ("base", "soup"):
            f, note = fine_fraction(preds2d[(kind, seed)], held2d)
            vals.append(f)
            notes.append(note)
        if vals[0] is None or vals[1] is None:
            print(f"      seed{seed}  spectral secondary DROPPED: {notes[0]}")
        else:
            print(f"      seed{seed}  energy finer than {FINE_SCALE_UM:.0f} um   "
                  f"base {vals[0]:.4f}   soup {vals[1]:.4f}   "
                  f"ratio {vals[1] / vals[0]:.4f}   ({notes[0]})")

    # ------------------------------------------------------------- bootstrap
    print("\n  BOOTSTRAP of the primary statistic (hardest quartile, FPR 5%, mean of seeds)")
    for block in (128, 256):
        res = bootstrap_primary(rows, cols, ink, strata_ink, preds, block, resamples, rng_seed)
        if res is None:
            print(f"    block {block:4d} px   no valid replicate")
            continue
        nb, lo, hi, pgt, nv = res
        print(f"    block {block:4d} px   n_blocks {nb:4d}   CI95 [{lo:+.4f}, {hi:+.4f}]   "
              f"P(delta>0) {pgt:.3f}   valid {nv}/{resamples}")
        if nb < 20:
            print(f"    {'':13s} WARNING: {nb} blocks is too few. Order of magnitude, not a bound.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--segment", default=None)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    segs = [a.segment] if a.segment else list(SEGMENTS)
    print(f"pitch {PITCH_UM} um/px   plane Z={PLANE}   strata {N_STRATA}   "
          f"primary FPR {100 * FPR_PRIMARY:.0f}%   resamples {a.resamples}")
    for s in segs:
        run_segment(a.labels_dir, a.out_dir, s, a.resamples, a.seed)


if __name__ == "__main__":
    main()
