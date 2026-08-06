"""Is `resolution_um` the raster pixel pitch, or just the name of the source scan?

The whole atlas rests on converting component size to microns, and that conversion uses the
number parsed out of the filename. If that number is the SCAN resolution rather than the
pitch the prediction was rendered at, every physical size in the atlas is wrong by whatever
factor separates the two - and the recipe comparison becomes an artefact of that error.

The test needs no metadata and no assumptions. Segments published under two recipes are the
same physical surface. So:

  width_px * pitch_um  must be equal for both renders.

If instead the two renders have the same width_px while declaring different resolutions,
then the declared resolution is not the pitch and the conversion is broken.
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    with open(args.index_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_seg = defaultdict(list)
    for r in rows:
        by_seg[(r["scroll"], r["segment"])].append(r)

    pairs = {k: v for k, v in by_seg.items() if len({r["recipe"] for r in v}) > 1}
    print(f"segments published under more than one recipe: {len(pairs)}")
    print(f"\n{'segment':<26}{'recipe':<11}{'decl um':>9}{'w_px':>8}{'h_px':>8}"
          f"{'w*um mm':>10}")

    agree_dims, agree_phys = 0, 0
    for (scroll, seg), rs in sorted(pairs.items()):
        got = []
        for r in sorted(rs, key=lambda r: r["recipe"]):
            path = os.path.join(args.cache_dir, r["key"].replace("/", "_"))
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            um = float(r["resolution_um"]) if r["resolution_um"] else 0.0
            h, w = img.shape
            got.append((r["recipe"], um, w, h, w * 8 * um / 1000.0))
        if len(got) < 2:
            continue
        for rec, um, w, h, mm in got:
            print(f"  {seg[:23]:<24}{rec:<11}{um:>9.3f}{w:>8}{h:>8}{mm:>10.1f}")
        a, b = got[0], got[1]
        rd = max(a[2], b[2]) / min(a[2], b[2])
        rp = max(a[4], b[4]) / min(a[4], b[4])
        print(f"  {'':<24}{'-> ratio':<11}{'':>9}{rd:>8.2f}{'':>8}{rp:>10.2f}")
        agree_dims += rd < 1.15
        agree_phys += rp < 1.15

    n = sum(1 for k, v in pairs.items() if len(v) >= 2)
    print(f"\npaired segments compared: {n}")
    print(f"  same PIXEL dimensions (ratio < 1.15): {agree_dims}")
    print(f"  same PHYSICAL width  (ratio < 1.15): {agree_phys}")
    print("\nIf pixel dimensions agree but physical widths do not, `resolution_um` names the")
    print("source scan and NOT the raster pitch, and the micron conversion is wrong.")


if __name__ == "__main__":
    main()
