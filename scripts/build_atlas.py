"""Score every published ink prediction for legibility, window by window.

Produces the failure atlas: a per-window table saying where the pipeline produced readable
text and where it produced speckle. Uses only already-published ds8 predictions, so it
costs bandwidth rather than GPU time.

Legibility is measured as structure at letter scale (see spatial_structure.py): a letter is
~1280 full-res px, so text concentrates its mass in large connected components while a
speckle field does not.
"""

import argparse
import csv
import os

import cv2
import numpy as np

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "vesuvius-challenge-open-data"
DS = 8  # the published downsample factor


def image_threshold(img):
    """Otsu over the on-segment pixels only.

    A fixed threshold is not comparable across recipes: PHerc0172's predictions are faint
    continuous tone while PHercParis4's are near-binary white, so at a fixed 128 the score
    measures prediction contrast rather than legibility. Off-segment area is exact zero and
    would drag Otsu's split down, so it is excluded.
    """
    on = img[img > 0]
    if on.size < 1000:
        return 128
    t, _ = cv2.threshold(on, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def score_window(win, thresh, um_per_px, big_um, huge_um, cap_um):
    """Legibility as a BAND on component size, measured in microns.

    Two things this has to get right that a naive version does not. First, the band: a lower
    bound alone cannot tell text from a saturated smear, because a page-sized white mass is
    one enormous connected component and scores as well as a page of letters. Second, the
    unit: a letter is a fixed PHYSICAL size (~3 mm), so a band in pixels silently asks for
    3x bigger letters at 7.91 um/px than at 2.4 um/px and fails the coarser scan by
    construction.

    `um_per_px` must be the pitch the prediction was RENDERED at, which is not the number in
    its filename - that names the source scan. See fetch_pitch.py.
    """
    binary = (win > thresh).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return {"frac_above": 0.0, "n_comp": 0, "median_extent": 0.0, "mass_big": 0.0,
                "mass_huge": 0.0, "mass_letter": 0.0, "mass_smear": 0.0}
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    extent = np.maximum(stats[1:, cv2.CC_STAT_WIDTH],
                        stats[1:, cv2.CC_STAT_HEIGHT]).astype(np.float64) * DS * um_per_px
    total = areas.sum()
    return {
        "frac_above": float(binary.mean()),
        "n_comp": int(n - 1),
        "median_extent": float(np.median(extent)),
        "mass_big": float(areas[extent >= big_um].sum() / total),
        "mass_huge": float(areas[extent >= huge_um].sum() / total),
        "mass_letter": float(areas[(extent >= big_um) & (extent <= cap_um)].sum() / total),
        "mass_smear": float(areas[extent > cap_um].sum() / total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True,
                    help="Output of fetch_pitch.py: the index plus a true pitch_um column")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    # Every published prediction gets a row here, scored or not. Without it a prediction that
    # no window can cover is silently absent from the atlas - in neither the numerator nor the
    # denominator - and any "% readable" is quoted over an unstated subset.
    ap.add_argument("--manifest-csv", required=True)
    # The window is a physical size, for the same reason the component band is. A fixed
    # 8192 full-res px covers 19.7 mm of papyrus at 2.4 um/px but 64.8 mm at 7.91 um/px -
    # eleven times the area. A bigger physical window is far more likely to straddle blank
    # or damaged papyrus, which dilutes a mass fraction, so a fixed pixel window quietly
    # penalises the coarsest scan. PHerc0172 averaged 1659 components per window against
    # ~110 elsewhere, which is that geometry and not its predictions.
    # The default is the 8192 px at 2.399 um/px that the reference zones were judged at.
    ap.add_argument("--window-um", type=float, default=19652.0)
    ap.add_argument("--stride-um", type=float, default=9826.0)
    ap.add_argument("--thresh", type=int, help="Fixed threshold; default is per-image Otsu")
    # Sizes in microns so they mean the same thing on a 1.1 um and a 7.9 um scan. Defaults
    # are the pixel cutoffs calibrated on the 2.4 um reference zones, converted across.
    ap.add_argument("--big-um", type=float, default=960)
    ap.add_argument("--huge-um", type=float, default=1920)
    ap.add_argument("--cap-um", type=float, default=7200,
                    help="Above this a component is a smear, not a letter (~2 letters wide)")
    ap.add_argument("--min-coverage", type=float, default=0.7,
                    help="Skip windows with more exact-zero (off-segment) area than this")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    with open(args.index_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    # No pitch, no score. Guessing it from the filename is what broke the first atlas: the
    # `1um_s1z2` predictions declare 1.129um but are rendered at 2.258um, so every physical
    # size for them came out 2x too small and the recipe comparison was decided by the bug.
    manifest = []
    missing = [r for r in rows if not r.get("pitch_um")]
    if missing:
        print(f"WARNING: {len(missing)} predictions have no resolved pitch and are skipped")
        for r in missing:
            manifest.append({"key": r["key"], "scroll": r["scroll"], "segment": r["segment"],
                             "recipe": r["recipe"], "pitch_um": "", "height_ds8": "",
                             "width_ds8": "", "window_px_ds8": "", "n_windows": 0,
                             "max_coverage": "", "status": "no_pitch"})
        rows = [r for r in rows if r.get("pitch_um")]

    out = open(args.out_csv, "w", newline="", encoding="utf-8")
    # `key` is what makes a row traceable to the prediction it came from. Without it,
    # (scroll, segment, recipe) is not unique - two models can render the same segment under
    # the same recipe label - and any join silently keeps whichever row came last.
    wr = csv.DictWriter(out, fieldnames=[
        "key",
        "scroll", "segment", "recipe", "resolution_um", "pitch_um", "window_px", "thresh",
        "y0", "x0", "coverage",
        "frac_above", "n_comp", "median_extent", "mass_big", "mass_huge",
        "mass_letter", "mass_smear"])
    wr.writeheader()

    n_win = 0
    for i, row in enumerate(rows, 1):
        rec = {"key": row["key"], "scroll": row["scroll"], "segment": row["segment"],
               "recipe": row["recipe"], "pitch_um": row["pitch_um"], "height_ds8": "",
               "width_ds8": "", "window_px_ds8": "", "n_windows": 0, "max_coverage": "",
               "status": ""}
        manifest.append(rec)
        local = os.path.join(args.cache_dir, row["key"].replace("/", "_"))
        if not os.path.exists(local):
            try:
                s3.download_file(BUCKET, row["key"], local)
            except Exception as e:
                print(f"  [{i}/{len(rows)}] download failed {row['segment']}: {e}")
                rec["status"] = "download_failed"
                continue
        img = cv2.imread(local, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [{i}/{len(rows)}] unreadable {local}")
            rec["status"] = "unreadable"
            continue
        h, w = img.shape
        thresh = args.thresh if args.thresh else image_threshold(img)
        um = float(row["pitch_um"])
        wpx = max(int(round(args.window_um / (um * DS))), 32)
        spx = max(int(round(args.stride_um / (um * DS))), 16)

        kept = 0
        n_geom = 0
        max_cov = 0.0
        for y in range(0, max(h - wpx, 0) + 1, spx):
            for x in range(0, max(w - wpx, 0) + 1, spx):
                win = img[y:y + wpx, x:x + wpx]
                if win.shape[0] < wpx // 2 or win.shape[1] < wpx // 2:
                    continue
                n_geom += 1
                coverage = float((win > 0).mean())
                max_cov = max(max_cov, coverage)
                if coverage < args.min_coverage:
                    continue
                s = score_window(win, thresh, um, args.big_um, args.huge_um, args.cap_um)
                wr.writerow({
                    "key": row["key"],
                    "scroll": row["scroll"], "segment": row["segment"],
                    "recipe": row["recipe"], "resolution_um": row["resolution_um"],
                    "pitch_um": um, "window_px": wpx * DS, "thresh": thresh,
                    "y0": y * DS, "x0": x * DS, "coverage": round(coverage, 3),
                    "frac_above": round(s["frac_above"], 4), "n_comp": s["n_comp"],
                    "median_extent": round(s["median_extent"], 1),
                    "mass_big": round(s["mass_big"], 4), "mass_huge": round(s["mass_huge"], 4),
                    "mass_letter": round(s["mass_letter"], 4),
                    "mass_smear": round(s["mass_smear"], 4),
                })
                kept += 1
        n_win += kept
        rec.update({"height_ds8": h, "width_ds8": w, "window_px_ds8": wpx, "n_windows": kept,
                    "max_coverage": round(max_cov, 3),
                    "status": "scored" if kept else
                              ("too_small" if n_geom == 0 else "low_coverage")})
        print(f"[{i}/{len(rows)}] {row['scroll']:<12} {row['segment'][:28]:<28} "
              f"{img.shape} pitch={um:<6} win={wpx * DS:<6} t={thresh:<4} windows={kept}")
        out.flush()

    out.close()
    with open(args.manifest_csv, "w", newline="", encoding="utf-8") as f:
        mw = csv.DictWriter(f, fieldnames=list(manifest[0]))
        mw.writeheader()
        mw.writerows(manifest)

    counts = {}
    for r in manifest:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nTOTAL windows scored: {n_win}")
    print(f"predictions: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"wrote {args.out_csv} and {args.manifest_csv}")


if __name__ == "__main__":
    main()
