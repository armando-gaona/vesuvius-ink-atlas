"""Does the metric itself lose points when the pixel gets coarser?

PHerc0172 is the only scroll in the corpus rendered at 7.91 um/px, and it is also the scroll
with the lowest legible fraction. That pairing is the most actionable line in the atlas - the
published literature measures ink detection degrading sharply above ~3.4 um/px, so the obvious
reading is "the scan is too coarse and it should be rescanned at 2.4".

It is also the fourth time this scroll has come last, and the first three were bugs of mine.

The confound is specific and has to be killed before the claim is published: every physical
size in the metric is expressed in microns, but that only makes it scale-INVARIANT if the
structure survives resampling. Otsu is recomputed per image, connected components merge and
break differently at coarser pitch, and a 2-pixel stroke at 7.91 um/px is a different object
than a 7-pixel stroke at 2.4. If our own scorer drops half its points when handed the same
papyrus at a coarser pitch, then PHerc0172's deficit is partly our artifact and the headline
collapses into the same family as findings 10b, 14 and 15.

So: take predictions that score well at 2.4 um/px, box-average them down to 7.91 um/px, rescore
them exactly as the atlas would, and compare.

WHAT THIS DOES NOT TEST, and must be stated wherever the result is quoted: this degrades the
MODEL OUTPUT, not the model input. A prediction actually computed from a 7.91 um scan had less
information at its input than this simulation ever removes. So the drop measured here is a
LOWER BOUND on the real effect, and it isolates exactly one term - the scale sensitivity of the
scorer. A small drop here means the metric is fair across pitches; it does not mean the
pipeline is fine.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_atlas import DS, image_threshold, score_window  # noqa: E402


def grid_scores(img, um, window_um, stride_um, min_coverage, big, huge, cap):
    """Score every window of one raster at the pitch it is claimed to be rendered at."""
    thresh = image_threshold(img)
    h, w = img.shape
    wpx = max(int(round(window_um / (um * DS))), 32)
    spx = max(int(round(stride_um / (um * DS))), 16)
    out = {}
    for iy, y in enumerate(range(0, max(h - wpx, 0) + 1, spx)):
        for ix, x in enumerate(range(0, max(w - wpx, 0) + 1, spx)):
            win = img[y:y + wpx, x:x + wpx]
            if win.shape[0] < wpx // 2 or win.shape[1] < wpx // 2:
                continue
            if float((win > 0).mean()) < min_coverage:
                continue
            # Keyed by grid index, not by pixel: the two rasters have different pixel sizes but
            # the same physical stride, so index (iy, ix) is the same patch of papyrus in both.
            out[(iy, ix)] = score_window(win, thresh, um, big, huge, cap)
    return out, thresh, wpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", default="data/predictions_pitch.csv")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-csv", default="data/pitch_ablation.csv")
    ap.add_argument("--from-pitch", type=float, default=2.399)
    ap.add_argument("--to-pitch", type=float, default=7.91)
    ap.add_argument("--scrolls", default="PHercParis4,PHerc1667")
    ap.add_argument("--n-preds", type=int, default=12)
    ap.add_argument("--window-um", type=float, default=19652.0)
    ap.add_argument("--stride-um", type=float, default=9826.0)
    ap.add_argument("--min-coverage", type=float, default=0.7)
    ap.add_argument("--big-um", type=float, default=960)
    ap.add_argument("--huge-um", type=float, default=1920)
    ap.add_argument("--cap-um", type=float, default=7200)
    ap.add_argument("--legible-at", type=float, default=0.900)
    args = ap.parse_args()

    scrolls = set(args.scrolls.split(","))
    with open(args.index_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["scroll"] in scrolls and r.get("pitch_um")
                and abs(float(r["pitch_um"]) - args.from_pitch) < 0.01]
    print(f"{len(rows)} predictions at {args.from_pitch} um/px in {sorted(scrolls)}")

    f = args.to_pitch / args.from_pitch
    print(f"simulating {args.to_pitch} um/px: box-average by {f:.3f}x\n")

    pairs = []
    done = 0
    for r in rows:
        if done >= args.n_preds:
            break
        local = os.path.join(args.cache_dir, r["key"].replace("/", "_"))
        img = cv2.imread(local, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        fine, t_fine, wpx_fine = grid_scores(img, args.from_pitch, args.window_um,
                                             args.stride_um, args.min_coverage,
                                             args.big_um, args.huge_um, args.cap_um)
        if len(fine) < 4:
            continue
        # INTER_AREA is a box average, which is what a detector with larger pixels does. Any
        # smoothing interpolation here would be flattering: it would preserve structure the
        # coarser sensor never recorded.
        small = cv2.resize(img, (max(int(round(img.shape[1] / f)), 1),
                                 max(int(round(img.shape[0] / f)), 1)),
                           interpolation=cv2.INTER_AREA)
        coarse, t_coarse, wpx_coarse = grid_scores(small, args.to_pitch, args.window_um,
                                                   args.stride_um, args.min_coverage,
                                                   args.big_um, args.huge_um, args.cap_um)
        shared = sorted(set(fine) & set(coarse))
        if not shared:
            continue
        for k in shared:
            pairs.append({
                "key": r["key"], "scroll": r["scroll"], "segment": r["segment"],
                "recipe": r["recipe"], "iy": k[0], "ix": k[1],
                "wpx_fine": wpx_fine, "wpx_coarse": wpx_coarse,
                "thresh_fine": t_fine, "thresh_coarse": t_coarse,
                "fine": round(fine[k]["mass_letter"], 4),
                "coarse": round(coarse[k]["mass_letter"], 4),
                "ncomp_fine": fine[k]["n_comp"], "ncomp_coarse": coarse[k]["n_comp"],
                "ext_fine": round(fine[k]["median_extent"], 1),
                "ext_coarse": round(coarse[k]["median_extent"], 1),
                "frac_fine": round(fine[k]["frac_above"], 4),
                "frac_coarse": round(coarse[k]["frac_above"], 4),
            })
        done += 1
        print(f"  {r['scroll']:<12} {r['segment']:<18} {len(shared):>4} paired windows "
              f"({wpx_fine}px -> {wpx_coarse}px, otsu {t_fine} -> {t_coarse})")

    if not pairs:
        print("no paired windows")
        return
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)

    a = np.array([p["fine"] for p in pairs])
    b = np.array([p["coarse"] for p in pairs])
    print(f"\n{len(pairs)} paired windows from {done} predictions\n")
    print(f"{'':<22}{'2.4 um':>10}{'7.91 um':>10}{'delta':>10}")
    for name, fn in (("mean mass_letter", np.mean), ("median", np.median),
                     ("p90", lambda v: np.percentile(v, 90))):
        print(f"{name:<22}{fn(a):>10.3f}{fn(b):>10.3f}{fn(b) - fn(a):>+10.3f}")
    la, lb = (a >= args.legible_at).mean(), (b >= args.legible_at).mean()
    print(f"{'frac >= ' + str(args.legible_at):<22}{la:>10.3f}{lb:>10.3f}{lb - la:>+10.3f}")
    print(f"\ncorrelation fine vs coarse: r = {np.corrcoef(a, b)[0, 1]:.3f}")

    # The headline is a threshold count, so the number that matters is not the mean shift but
    # how many windows change SIDE. A metric can move every score a little and flip nothing.
    lost = int(((a >= args.legible_at) & (b < args.legible_at)).sum())
    gained = int(((a < args.legible_at) & (b >= args.legible_at)).sum())
    n_leg = int((a >= args.legible_at).sum())
    print(f"legible at 2.4 um : {n_leg}")
    print(f"  lost by coarsening : {lost}"
          f"{f'  ({lost / n_leg:.1%} of them)' if n_leg else ''}")
    print(f"  gained             : {gained}")

    nc = np.array([p["ncomp_fine"] for p in pairs]), np.array([p["ncomp_coarse"] for p in pairs])
    ex = np.array([p["ext_fine"] for p in pairs]), np.array([p["ext_coarse"] for p in pairs])
    print(f"\nmedian n_comp   {np.median(nc[0]):>8.0f}{np.median(nc[1]):>10.0f}")
    print(f"median extent   {np.median(ex[0]):>8.0f}{np.median(ex[1]):>10.0f}  um")


if __name__ == "__main__":
    main()
