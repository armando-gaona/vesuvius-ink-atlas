"""Recompute every aggregate in `results/` in a single pass, at full precision.

    python scripts/recompute_tables.py soup    --pred-dir DIR --labels-dir DIR
    python scripts/recompute_tables.py steps   --pred-dir DIR --labels-dir DIR
    python scripts/recompute_tables.py zwindow --pred-dir DIR --labels-dir DIR

Why this exists. The first version of the results tables was built by chaining values that had
already been rounded to four decimals: a mean of two AUCs was computed from the printed 0.8683 and
0.8433 rather than from the full-precision numbers. That is invisible most of the time and then it
is not, and the same quantity ended up printed as 0.0145 in one document and 0.0146 in another.
Here every AUC is recomputed from the TIFFs in one pass and rounded only when printed, so the
tables can be regenerated instead of transcribed.

Each subcommand writes a JSON with full-precision values next to the printed report, and aborts up
front if any input TIFF is missing rather than producing a partial table.

Expected file names in `--pred-dir` (`TAG` is the segment name, except `pherc0814-46527` whose
files were named `0814`):

    {SEGMENT}_{seed}_s75k.tif          the published step-075000 baseline
    soup_{TAG}_A4_{seed}_f64.tif       the soup of the last 4 steps
    sweep_{TAG}_{seed}_s020000.tif     the step-020000 checkpoint
    zwin_{TAG}_{seed}_{S0..S4}.tif     the five z window positions

`--labels-dir` holds one subdirectory per segment with the published
`{SEGMENT}_{inklabels,supervision_mask,validation_mask}.zarr`.
"""
import argparse
import json
import os

import numpy as np
import tifffile
import zarr
from scipy.stats import rankdata

PLANE = 10
SEEDS = ("seed42", "seed43")
SEGMENTS = ("pherc0814-46527", "pherc0139-w016", "pherc1667-w029")
SHORT = {"pherc0814-46527": "0814"}
SELECTION = "pherc0814-46527"  # the segment every choice was made on


def tag(seg):
    return SHORT.get(seg, seg)


def to_unit(q):
    """Normalise by DTYPE, never by the observed maximum: the scale must not depend on the data."""
    if q.dtype == np.uint8:
        return q.astype(np.float32) / 255.0
    if q.dtype == np.uint16:
        return q.astype(np.float32) / 65535.0
    if q.dtype in (np.float32, np.float64):
        return q.astype(np.float32)
    raise SystemExit(f"unsupported dtype {q.dtype}: the scale would have to be guessed")


def load(path, shape):
    q = to_unit(tifffile.imread(path))
    if q.shape != shape:
        raise SystemExit(f"shape mismatch on {path}: comparing it would be an alignment bug")
    return q


def require(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))


def regions(labels_dir, seg):
    def op(name):
        return zarr.open(os.path.join(labels_dir, seg, f"{seg}_{name}.zarr", "0"), mode="r")
    ink = op("inklabels")
    shape = tuple(ink.shape[1:])
    y = np.asarray(ink[PLANE]) > 0
    m = np.asarray(op("supervision_mask")[PLANE]) > 0
    v = np.asarray(op("validation_mask")[PLANE]) > 0
    # The two masks are disjoint, not nested, so each region is taken as it comes.
    return shape, y, m & ~v, v & ~m


def auc(scores, labels):
    npos = int(labels.sum())
    nneg = labels.size - npos
    if npos == 0 or nneg == 0:
        raise SystemExit("only one class present in the region: no AUC is defined")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def bacc(scores, labels, th=0.50):
    """Balanced accuracy at 0.50, which is `best_checkpoint_metric` in the checkpoints' config."""
    pr = scores > th
    sens = float((pr & labels).sum()) / int(labels.sum())
    spec = float((~pr & ~labels).sum()) / int((~labels).sum())
    return 0.5 * (sens + spec)


# --------------------------------------------------------------------------- soup

def run_soup(pred, labels_dir):
    def base_tif(seg, sd):
        return os.path.join(pred, f"{seg}_{sd}_s75k.tif")

    def a4_tif(seg, sd):
        # ALWAYS the _f64 variant: it is the one the published weight_soup.py produces
        return os.path.join(pred, f"soup_{tag(seg)}_A4_{sd}_f64.tif")

    require([f(s, sd) for s in SEGMENTS for sd in SEEDS for f in (base_tif, a4_tif)])

    res = {}
    for seg in SEGMENTS:
        shape, y, train, held = regions(labels_dir, seg)
        r, preds = {}, {}
        for sd in SEEDS:
            for cfg, path in (("base", base_tif(seg, sd)), ("A4", a4_tif(seg, sd))):
                p = load(path, shape)
                preds[(cfg, sd)] = p
                r[f"{cfg}_{sd}"] = {
                    "held_auc": auc(p[held], y[held]),
                    "supervised_auc": auc(p[train], y[train]),
                    "held_bacc": bacc(p[held], y[held]),
                    "file": os.path.basename(path),
                }
        for cfg in ("base", "A4"):
            # the TWO-inference tier: AUC of the AVERAGED PREDICTION, a different quantity
            # from the mean of two AUCs below, and not comparable with it
            avg = 0.5 * (preds[(cfg, "seed42")] + preds[(cfg, "seed43")])
            r[f"{cfg}_predavg_held_auc"] = auc(avg[held], y[held])
            r[f"{cfg}_mean_auc"] = 0.5 * (r[f"{cfg}_seed42"]["held_auc"]
                                          + r[f"{cfg}_seed43"]["held_auc"])
            r[f"{cfg}_mean_bacc"] = 0.5 * (r[f"{cfg}_seed42"]["held_bacc"]
                                           + r[f"{cfg}_seed43"]["held_bacc"])
            r[f"{cfg}_seed_separation"] = abs(r[f"{cfg}_seed42"]["held_auc"]
                                              - r[f"{cfg}_seed43"]["held_auc"])
        r["delta_mean_auc"] = r["A4_mean_auc"] - r["base_mean_auc"]
        r["delta_mean_bacc"] = r["A4_mean_bacc"] - r["base_mean_bacc"]
        r["delta_predavg"] = r["A4_predavg_held_auc"] - r["base_predavg_held_auc"]
        res[seg] = r

        print(f"\n=== {seg}")
        for sd in SEEDS:
            b, a = r[f"base_{sd}"], r[f"A4_{sd}"]
            print(f"  {sd}  held out  {b['held_auc']:.6f} -> {a['held_auc']:.6f}  "
                  f"delta {a['held_auc'] - b['held_auc']:+.6f}")
            print(f"          supervised {b['supervised_auc']:.6f} -> {a['supervised_auc']:.6f}")
            print(f"          bal. acc.  {b['held_bacc']:.6f} -> {a['held_bacc']:.6f}  "
                  f"delta {a['held_bacc'] - b['held_bacc']:+.6f}")
        print(f"  mean of the two AUCs      {r['base_mean_auc']:.6f} -> {r['A4_mean_auc']:.6f}"
              f"  delta {r['delta_mean_auc']:+.6f}")
        print(f"  mean balanced accuracy    {r['base_mean_bacc']:.6f} -> {r['A4_mean_bacc']:.6f}"
              f"  delta {r['delta_mean_bacc']:+.6f}")
        print(f"  seed separation           {r['base_seed_separation']:.6f} -> "
              f"{r['A4_seed_separation']:.6f}")
        print(f"  AUC of the averaged prediction (2 inferences) "
              f"{r['base_predavg_held_auc']:.6f} -> {r['A4_predavg_held_auc']:.6f}"
              f"  delta {r['delta_predavg']:+.6f}")

    tr = [res[s][f"A4_{sd}"]["supervised_auc"] for s in SEGMENTS for sd in SEEDS]
    print(f"\nA4 supervised AUC over the 6 runs: min {min(tr):.6f}  max {max(tr):.6f}"
          "   (the mandatory control: the soup still works as a model)")
    print("mean seed separation: "
          f"{np.mean([res[s]['base_seed_separation'] for s in SEGMENTS]):.6f} -> "
          f"{np.mean([res[s]['A4_seed_separation'] for s in SEGMENTS]):.6f}")
    return res


# --------------------------------------------------------------------------- steps

def run_steps(pred, labels_dir):
    def tif(seg, step, sd):
        if step == "20k":
            return os.path.join(pred, f"sweep_{tag(seg)}_{sd}_s020000.tif")
        return os.path.join(pred, f"{seg}_{sd}_s75k.tif")

    require([tif(s, st, sd) for s in SEGMENTS for st in ("20k", "75k") for sd in SEEDS])

    res = {}
    for seg in SEGMENTS:
        shape, y, train, held = regions(labels_dir, seg)
        r = {}
        for step in ("20k", "75k"):
            preds = {}
            for sd in SEEDS:
                p = load(tif(seg, step, sd), shape)
                preds[sd] = p
                r[f"{step}_{sd}_held"] = auc(p[held], y[held])
                r[f"{step}_{sd}_supervised"] = auc(p[train], y[train])
            r[f"{step}_sep_held"] = abs(r[f"{step}_seed42_held"] - r[f"{step}_seed43_held"])
            r[f"{step}_sep_supervised"] = abs(r[f"{step}_seed42_supervised"]
                                              - r[f"{step}_seed43_supervised"])
            r[f"{step}_mean_held"] = 0.5 * (r[f"{step}_seed42_held"] + r[f"{step}_seed43_held"])
            avg = 0.5 * (preds["seed42"] + preds["seed43"])
            r[f"{step}_predavg_held"] = auc(avg[held], y[held])
        res[seg] = r

        print(f"\n=== {seg}")
        for step in ("20k", "75k"):
            print(f"  {step}  held out s42 {r[f'{step}_seed42_held']:.6f}  "
                  f"s43 {r[f'{step}_seed43_held']:.6f}  "
                  f"mean of the two AUCs {r[f'{step}_mean_held']:.6f}  "
                  f"averaged prediction {r[f'{step}_predavg_held']:.6f}")
            print(f"        seed separation: held out {r[f'{step}_sep_held']:.6f}   "
                  f"supervised {r[f'{step}_sep_supervised']:.6f}")

    print()
    for step in ("20k", "75k"):
        for zone in ("held", "supervised"):
            vals = [res[s][f"{step}_sep_{zone}"] for s in SEGMENTS]
            print(f"mean seed separation, {step}, {zone}: {np.mean(vals):.6f}")
    return res


# ------------------------------------------------------------------------- zwindow

def run_zwindow(pred, labels_dir):
    positions = ("S0", "S1", "S2", "S3", "S4")

    def zwin(seg, sd, pos):
        return os.path.join(pred, f"zwin_{tag(seg)}_{sd}_{pos}.tif")

    def base(seg, sd):  # S2 is the default window, already measured as the 75k baseline
        return os.path.join(pred, f"{seg}_{sd}_s75k.tif")

    require([zwin(SELECTION, sd, p) for sd in SEEDS for p in positions]
            + [base(s, sd) for s in SEGMENTS for sd in SEEDS]
            + [zwin(s, sd, "S3") for s in SEGMENTS for sd in SEEDS])

    res = {}

    # --- stage 1: the five positions on the selection segment, plus the two pre-registered
    #     ensembles. They are all the windows that fit (17 planes inside 21), not a chosen grid.
    shape, y, train, held = regions(labels_dir, SELECTION)
    preds = {(p, sd): load(zwin(SELECTION, sd, p), shape) for p in positions for sd in SEEDS}
    stage1 = {}
    for pos in positions:
        r = {}
        for sd in SEEDS:
            r[f"held_{sd}"] = auc(preds[(pos, sd)][held], y[held])
            r[f"supervised_{sd}"] = auc(preds[(pos, sd)][train], y[train])
        r["held_mean"] = 0.5 * (r["held_seed42"] + r["held_seed43"])
        stage1[pos] = r
    for name, members in (("E3", ("S1", "S2", "S3")), ("E5", positions)):
        r = {}
        for sd in SEEDS:
            avg = sum(preds[(m, sd)] for m in members) / len(members)
            r[f"held_{sd}"] = auc(avg[held], y[held])
        r["held_mean"] = 0.5 * (r["held_seed42"] + r["held_seed43"])
        stage1[name] = r
    res["stage1"] = stage1

    print(f"=== stage 1, {SELECTION}")
    for k, r in stage1.items():
        sup = (f"   supervised {r['supervised_seed42']:.6f}/{r['supervised_seed43']:.6f}"
               if "supervised_seed42" in r else "")
        print(f"  {k:3s}  held out s42 {r['held_seed42']:.6f}  s43 {r['held_seed43']:.6f}  "
              f"mean {r['held_mean']:.6f}{sup}")
    sup_all = [stage1[p][f"supervised_{sd}"] for p in positions for sd in SEEDS]
    print(f"  supervised: min {min(sup_all):.6f}  max {max(sup_all):.6f}  "
          f"range {max(sup_all) - min(sup_all):.6f}   (the mandatory control)")
    for label, key in (("s42", "held_seed42"), ("s43", "held_seed43"), ("mean", "held_mean")):
        vals = [stage1[p][key] for p in positions]
        print(f"  held-out range, {label}: {max(vals) - min(vals):.6f}")

    # --- stage 2: S3 on the other two segments, without re-selecting anything
    print("\n=== stage 2")
    stage2 = {}
    for seg in SEGMENTS:
        shape, y, train, held = regions(labels_dir, seg)
        r = {}
        for sd in SEEDS:
            r[f"S2_{sd}"] = auc(load(base(seg, sd), shape)[held], y[held])
            r[f"S3_{sd}"] = auc(load(zwin(seg, sd, "S3"), shape)[held], y[held])
        r["S2_mean"] = 0.5 * (r["S2_seed42"] + r["S2_seed43"])
        r["S3_mean"] = 0.5 * (r["S3_seed42"] + r["S3_seed43"])
        r["delta"] = r["S3_mean"] - r["S2_mean"]
        stage2[seg] = r
        print(f"  {seg:18s} S2 {r['S2_mean']:.6f} -> S3 {r['S3_mean']:.6f}  "
              f"delta {r['delta']:+.6f}")
        for sd in SEEDS:
            print(f"      {sd}  {r[f'S2_{sd}']:.6f} -> {r[f'S3_{sd}']:.6f}  "
                  f"{r[f'S3_{sd}'] - r[f'S2_{sd}']:+.6f}")
    res["stage2"] = stage2
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=("soup", "steps", "zwindow"))
    ap.add_argument("--pred-dir", required=True, help="directory holding the prediction TIFFs")
    ap.add_argument("--labels-dir", required=True, help="directory holding the label zarrs")
    ap.add_argument("--json", help="where to write the full-precision values")
    a = ap.parse_args()

    res = {"soup": run_soup, "steps": run_steps, "zwindow": run_zwindow}[a.experiment](
        a.pred_dir, a.labels_dir)

    dst = a.json or f"{a.experiment}_tables_full_precision.json"
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
