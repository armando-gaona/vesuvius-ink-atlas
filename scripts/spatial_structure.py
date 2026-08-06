"""Measure whether an ink prediction has structure at letter scale.

Intensity statistics rank our three reference zones only weakly, because a speckle field
and a page of text can carry the same amount of high-confidence mass. What separates them
by eye is SCALE: a letter is a connected blob ~1280 full-res px across, speckle is a cloud
of blobs 10-50x smaller. This measures that directly.

Operates on prediction images (official reference or our own), not on raw volumes.
"""

import argparse

import cv2
import numpy as np


def structure_signals(img, ds, thresh):
    """img: grayscale prediction. ds: downsample factor vs full-res. Sizes reported in
    full-res pixels so they can be compared against letter size (~1280 px)."""
    binary = (img > thresh).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) if n > 1 else np.array([0.0])
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64) if n > 1 else np.array([0.0])
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64) if n > 1 else np.array([0.0])
    extent = np.maximum(widths, heights) * ds  # full-res px

    total = areas.sum()
    # Fraction of ink mass living in blobs at least a third of a letter across. Text puts
    # most of its mass in big components; speckle puts almost none.
    big = float(areas[extent >= 400].sum() / total) if total > 0 else 0.0
    huge = float(areas[extent >= 800].sum() / total) if total > 0 else 0.0

    return {
        "frac_above_thresh": round(float(binary.mean()), 4),
        "n_components": int(n - 1),
        "median_extent_px": round(float(np.median(extent)), 1),
        "p90_extent_px": round(float(np.percentile(extent, 90)), 1),
        "max_extent_px": round(float(extent.max()), 1),
        "mass_in_blobs_ge400px": round(big, 4),
        "mass_in_blobs_ge800px": round(huge, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--thresh", type=int, default=128)
    ap.add_argument("--zone", nargs=4, metavar=("LABEL", "Y0", "X0", "SIZE"),
                    action="append", required=True, help="Full-res coords")
    args = ap.parse_args()

    ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    rows = []
    for label, y0, x0, size in args.zone:
        y0, x0, size = int(y0), int(x0), int(size)
        ry, rx, rs = y0 // args.ds, x0 // args.ds, size // args.ds
        sub = ref[ry:ry + rs, rx:rx + rs]
        s = structure_signals(sub, args.ds, args.thresh)
        s["label"] = label
        rows.append(s)

    keys = ["frac_above_thresh", "n_components", "median_extent_px", "p90_extent_px",
            "max_extent_px", "mass_in_blobs_ge400px", "mass_in_blobs_ge800px"]
    print(f"{'signal':<24}" + "".join(f"{r['label']:>14}" for r in rows))
    for k in keys:
        print(f"{k:<24}" + "".join(f"{r[k]:>14}" for r in rows))


if __name__ == "__main__":
    main()
