"""Is the weight soup better than a well chosen single checkpoint, or only better than the last one?

    python scripts/checkpoint_baseline.py <labels_dir> <out_dir>

`soup_results.md` compares A4 against `step-075000`, the final published checkpoint, and reports
+0.010 to +0.019 of held-out AUC on three segments. That is the checkpoint anyone downloads, so it
is a fair baseline for the question "should I soup what I already have". It is not a fair baseline
for the question a reviewer actually asks, which is whether the soup beats the *family* it averages.

`step_sweep_results.md` already measured step 20k on the same three segments, the same held-out
region, the same plane and the same metric. The two documents were never put side by side. This
script does that, recomputing every number from the TIFFs rather than chaining values that were
already rounded for printing, so the table here cannot drift away from the images it describes.

The comparison is symmetric in its procedure, which is what makes it worth reporting: A4 and step
20k were both selected on `pherc0814-46527` and then applied unchanged to the other two segments.
Neither got to re-select. So whatever selection advantage exists, both carry it equally.

The gate runs first. Several soup builds sit on disk (float32 and float64 accumulations of the same
soup) and silently picking the wrong file would answer a different question.
"""
import os
import sys

import numpy as np
import tifffile
import zarr
from scipy.stats import rankdata

PLANE = 10
SEEDS = (42, 43)
GATE_TOL = 5e-4

# (segment, short tag used in the sweep and soup filenames)
SEGMENTS = (
    ("pherc0814-46527", "0814"),
    ("pherc0139-w016", "pherc0139-w016"),
    ("pherc1667-w029", "pherc1667-w029"),
)

# Published held-out AUCs per seed. Baselines from holdout9_results.md, A4 from soup_results.md.
PUBLISHED = {
    "pherc0814-46527": {"base": (0.8683, 0.8433), "soup": (0.8692, 0.8630)},
    "pherc0139-w016":  {"base": (0.8014, 0.9087), "soup": (0.8526, 0.8960)},
    "pherc1667-w029":  {"base": (0.8717, 0.9197), "soup": (0.8966, 0.9297)},
}

# Published step-sweep AUCs on pherc0814-46527, stage 1 table. Used as a second gate: this script
# claims to reproduce that table too, so it should be made to prove it.
PUBLISHED_SWEEP_0814 = {
    "010000": (0.8760, 0.8697),
    "020000": (0.8834, 0.8931),
    "030000": (0.8476, 0.8893),
    "040000": (0.8352, 0.8523),
    "050000": (0.8910, 0.8427),
    "060000": (0.8511, 0.8426),
    "075000": (0.8683, 0.8433),
}

SWEEP_STEPS = ("010000", "020000", "030000", "040000", "050000", "060000", "075000")


def to_unit(q):
    if q.dtype == np.uint8:
        return q.astype(np.float32) / 255.0
    if q.dtype == np.uint16:
        return q.astype(np.float32) / 65535.0
    if q.dtype in (np.float32, np.float64):
        return q.astype(np.float32)
    raise SystemExit(f"unsupported dtype {q.dtype}: the scale would have to be guessed")


def auc(scores, labels):
    npos = int(labels.sum())
    nneg = labels.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def held_out(labels_dir, segment):
    def open_(name):
        return zarr.open(os.path.join(labels_dir, segment, f"{segment}_{name}.zarr", "0"), mode="r")
    ink = open_("inklabels")
    y = np.asarray(ink[PLANE]) > 0
    m = np.asarray(open_("supervision_mask")[PLANE]) > 0
    v = np.asarray(open_("validation_mask")[PLANE]) > 0
    return y, v & ~m, tuple(ink.shape[1:])


def load(path, shape):
    if not os.path.exists(path):
        raise SystemExit(f"missing prediction: {path}")
    q = to_unit(tifffile.imread(path))
    if q.shape != shape:
        raise SystemExit(f"shape mismatch for {path}: {q.shape} vs {shape}")
    return q


def paths_for(out_dir, segment, tag, seed):
    base = os.path.join(out_dir, f"{segment}_seed{seed}_s75k.tif")
    soup = os.path.join(out_dir, f"soup_{tag}_A4_seed{seed}_f64.tif")
    step20k = os.path.join(out_dir, f"sweep_{tag}_seed{seed}_s020000.tif")
    return base, soup, step20k


def sweep_path(out_dir, segment, tag, seed, step):
    if step == "075000":
        return os.path.join(out_dir, f"{segment}_seed{seed}_s75k.tif")
    return os.path.join(out_dir, f"sweep_{tag}_seed{seed}_s{step}.tif")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    labels_dir, out_dir = sys.argv[1], sys.argv[2]

    scores = {}
    print("=" * 78)
    print("GATE: reproduce the published held-out AUCs before computing anything new")
    print("=" * 78)
    failures = []
    for segment, tag in SEGMENTS:
        y, held, shape = held_out(labels_dir, segment)
        yy = y[held]
        for si, seed in enumerate(SEEDS):
            base_p, soup_p, s20_p = paths_for(out_dir, segment, tag, seed)
            a_base = auc(load(base_p, shape)[held], yy)
            a_soup = auc(load(soup_p, shape)[held], yy)
            a_20k = auc(load(s20_p, shape)[held], yy)
            scores[(segment, seed)] = {"base": a_base, "soup": a_soup, "20k": a_20k}
            for what, got in (("base", a_base), ("soup", a_soup)):
                want = PUBLISHED[segment][what][si]
                ok = abs(got - want) <= GATE_TOL
                print(f"  {segment:16s} seed{seed} {what:5s} got {got:.4f}  published {want:.4f}  "
                      f"{'ok' if ok else 'MISMATCH'}")
                if not ok:
                    failures.append((segment, seed, what, got, want))
        print(f"  {segment:16s} held-out pixels {yy.size:,}  ink {100.0 * yy.mean():.2f}%")
    if failures:
        raise SystemExit(f"\nGATE FAILED on {len(failures)} cell(s). Everything below would answer "
                         "a different question, so nothing is computed.")
    print("\nGate passed. The files on disk are the ones the published tables describe.\n")

    print("=" * 78)
    print("HEAD TO HEAD: final checkpoint, soup A4, and single checkpoint step-020000")
    print("=" * 78)
    print(f"{'segment':18s} {'seed':>5s} {'75k':>8s} {'A4 soup':>8s} {'20k':>8s} "
          f"{'20k-soup':>9s}")
    per_segment = {}
    for segment, _ in SEGMENTS:
        vals = {"base": [], "soup": [], "20k": []}
        for seed in SEEDS:
            s = scores[(segment, seed)]
            for k in vals:
                vals[k].append(s[k])
            print(f"{segment:18s} {seed:5d} {s['base']:8.4f} {s['soup']:8.4f} {s['20k']:8.4f} "
                  f"{s['20k'] - s['soup']:+9.4f}")
        per_segment[segment] = {k: float(np.mean(v)) for k, v in vals.items()}
    print()
    print(f"{'segment':18s} {'75k':>8s} {'A4 soup':>8s} {'20k':>8s} {'soup-75k':>9s} "
          f"{'20k-75k':>9s} {'20k-soup':>9s}")
    for segment, _ in SEGMENTS:
        m = per_segment[segment]
        print(f"{segment:18s} {m['base']:8.4f} {m['soup']:8.4f} {m['20k']:8.4f} "
              f"{m['soup'] - m['base']:+9.4f} {m['20k'] - m['base']:+9.4f} "
              f"{m['20k'] - m['soup']:+9.4f}")
    gm = {k: float(np.mean([per_segment[s][k] for s, _ in SEGMENTS])) for k in ("base", "soup", "20k")}
    print(f"{'MEAN':18s} {gm['base']:8.4f} {gm['soup']:8.4f} {gm['20k']:8.4f} "
          f"{gm['soup'] - gm['base']:+9.4f} {gm['20k'] - gm['base']:+9.4f} "
          f"{gm['20k'] - gm['soup']:+9.4f}")

    wins_20k = sum(1 for s, _ in SEGMENTS if per_segment[s]["20k"] > per_segment[s]["soup"])
    print(f"\nstep-020000 beats the soup on {wins_20k} of {len(SEGMENTS)} segments "
          f"(segment means, mean of the two seeds).")

    print()
    print("=" * 78)
    print("WHERE THE SOUP SITS INSIDE THE FAMILY IT AVERAGES (pherc0814-46527 only)")
    print("=" * 78)
    print("The full seven-step sweep exists on disk only for this segment. A4 averages the last")
    print("four steps (40k, 50k, 60k, 75k), so the family it is drawn from is those four; the")
    print("earlier three are shown because the question is what a user could have picked.")
    segment, tag = SEGMENTS[0]
    y, held, shape = held_out(labels_dir, segment)
    yy = y[held]
    sweep = {}
    gate_fail = []
    for step in SWEEP_STEPS:
        row = []
        for si, seed in enumerate(SEEDS):
            a_ = auc(load(sweep_path(out_dir, segment, tag, seed, step), shape)[held], yy)
            row.append(a_)
            want = PUBLISHED_SWEEP_0814[step][si]
            if abs(a_ - want) > GATE_TOL:
                gate_fail.append((step, seed, a_, want))
        sweep[step] = row
    if gate_fail:
        for step, seed, got, want in gate_fail:
            print(f"  MISMATCH step {step} seed{seed}: got {got:.4f} published {want:.4f}")
        raise SystemExit("the step sweep does not reproduce; the table below would be untrustworthy")
    print("(the stage 1 table of step_sweep_results.md reproduces exactly)\n")
    print(f"{'step':>8s} {'seed42':>8s} {'seed43':>8s} {'mean':>8s}   {'in A4':>5s}")
    for step in SWEEP_STEPS:
        r = sweep[step]
        inA4 = "yes" if step in ("040000", "050000", "060000", "075000") else "no"
        print(f"{step:>8s} {r[0]:8.4f} {r[1]:8.4f} {float(np.mean(r)):8.4f}   {inA4:>5s}")
    soup_row = [scores[(segment, s)]["soup"] for s in SEEDS]
    print(f"{'A4 soup':>8s} {soup_row[0]:8.4f} {soup_row[1]:8.4f} {float(np.mean(soup_row)):8.4f}")
    for si, seed in enumerate(SEEDS):
        better = [st for st in SWEEP_STEPS if sweep[st][si] > soup_row[si]]
        print(f"  seed{seed}: {len(better)} of {len(SWEEP_STEPS)} individual checkpoints beat the "
              f"soup ({', '.join(better) if better else 'none'})")
    within = [st for st in ("040000", "050000", "060000", "075000")]
    for si, seed in enumerate(SEEDS):
        better = [st for st in within if sweep[st][si] > soup_row[si]]
        print(f"  seed{seed}: {len(better)} of 4 checkpoints INSIDE the soup beat the soup "
              f"({', '.join(better) if better else 'none'})")


if __name__ == "__main__":
    main()
