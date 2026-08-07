"""Summarise the failure atlas and render contact sheets of its extremes.

The contact sheets are the validation step: if the top-scoring windows are not visibly
readable and the bottom-scoring ones are not visibly speckle, the metric is wrong and the
atlas means nothing. Rendered with the score burned in so they can be checked one by one.
"""

import argparse
import csv
import os

import cv2
import numpy as np

DS = 8


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("y0", "x0", "n_comp"):
            r[k] = int(r[k])
        for k in ("coverage", "frac_above", "median_extent", "mass_big", "mass_huge",
                  "mass_letter", "mass_smear"):
            r[k] = float(r[k])
    return rows


def tiles_of(rows, index, cache_dir, cell, score):
    tiles = []
    for r in rows:
        # The atlas row carries the S3 key of the prediction it came from. Looking it up by
        # (scroll, segment, recipe) instead used to crop from whichever of two predictions
        # published under that label happened to be indexed first.
        local = os.path.join(cache_dir, r["key"].replace("/", "_"))
        img = cv2.imread(local, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Windows are a fixed PHYSICAL size, so their pixel size varies with the pitch the
        # prediction was rendered at. Cropping a constant number of pixels here would show a
        # different region than the one that was scored.
        wpx = int(r["window_px"]) // DS
        y, x = r["y0"] // DS, r["x0"] // DS
        win = img[y:y + wpx, x:x + wpx]
        if win.size == 0:
            continue
        win = cv2.resize(win, (cell, cell), interpolation=cv2.INTER_AREA)
        win = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(win, (0, 0), (cell - 1, 18), (0, 0, 0), -1)
        cv2.putText(win, f"{r[score]:.2f} {r['scroll'][:9]}",
                    (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
        cv2.rectangle(win, (0, 0), (cell - 1, cell - 1), (60, 60, 60), 1)
        tiles.append(win)
    return tiles


def contact_sheet(rows, index, cache_dir, cols, cell, out_path, score):
    tiles = tiles_of(rows, index, cache_dir, cell, score)
    if not tiles:
        print(f"no tiles for {out_path}")
        return
    rows_n = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows_n * cols:
        tiles.append(np.zeros((cell, cell, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i * cols:(i + 1) * cols]) for i in range(rows_n)])
    cv2.imwrite(out_path, sheet)
    print(f"wrote {out_path}  ({len(rows)} windows)")


def ladder(rows, index, cache_dir, per_band, cell, out_path, score, seed=0):
    """One row per score decile: the artifact that turns a ranking into a threshold.

    The contact sheets only prove the extremes are ordered correctly. They say nothing about
    WHERE along the range text actually appears, and without that the score cannot be used
    to call a window legible or not.
    """
    rng = np.random.default_rng(seed)
    ordered = sorted(rows, key=lambda r: r[score])
    bands = np.array_split(np.arange(len(ordered)), 10)

    strips = []
    for band in reversed(bands):
        pick = rng.choice(band, size=min(per_band, len(band)), replace=False)
        sel = [ordered[i] for i in sorted(pick)]
        lo, hi = ordered[band[0]][score], ordered[band[-1]][score]
        strip = tiles_of(sel, index, cache_dir, cell, score)
        while len(strip) < per_band:
            strip.append(np.zeros((cell, cell, 3), np.uint8))
        label = np.zeros((cell, 90, 3), np.uint8)
        cv2.putText(label, f"{lo:.2f}-{hi:.2f}", (4, cell // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        strips.append(np.hstack([label] + strip))

    cv2.imwrite(out_path, np.vstack(strips))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-csv", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--cell", type=int, default=190)
    ap.add_argument("--score", default="mass_letter", help="Column to rank windows by")
    ap.add_argument("--scroll", help="Restrict the report to a single scroll")
    ap.add_argument("--legible-at", type=float, default=0.900,
                    help="Calibrated legibility threshold (see calibrate.py)")
    ap.add_argument("--ink-floor", type=float, default=0.05,
                    help="Score 0 below this frac_above. 0 disables. See calibrate.py.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load(args.atlas_csv)
    for r in rows:
        if r["frac_above"] < args.ink_floor:
            r[args.score] = 0.0
    if args.scroll:
        rows = [r for r in rows if r["scroll"] == args.scroll]
    # Percentages go over a non-overlapping tiling; the sheets still draw on every window.
    # The atlas strides by half a window, so a percentage over the full grid counts each
    # patch of papyrus about four times and is a percentage of nothing physical.
    stat = [r for r in rows
            if (r["y0"] // (int(r["window_px"]) // 2)) % 2 == 0
            and (r["x0"] // (int(r["window_px"]) // 2)) % 2 == 0]
    print(f"windows: {len(rows)}  non-overlapping: {len(stat)}  "
          f"segments: {len({r['segment'] for r in rows})}")

    mh = np.array([r[args.score] for r in stat])
    fa = np.array([r["frac_above"] for r in stat])
    me = np.array([r["median_extent"] for r in stat])

    print(f"\n{args.score} distribution:")
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  p{q:<3} {np.percentile(mh, q):.4f}")
    print(f"\nfrac(score == 0)   : {float((mh == 0).mean()):.3f}")
    print(f"frac(score >= 0.5) : {float((mh >= 0.5).mean()):.3f}")

    # If legibility were just 'amount of ink', the score would track frac_above and add
    # nothing over a simple brightness threshold.
    print(f"\ncorr(score, frac_above)    : {np.corrcoef(mh, fa)[0, 1]:.3f}")
    print(f"corr(score, median_extent): {np.corrcoef(mh, me)[0, 1]:.3f}")

    # The threshold is anchored on 125 blind stratified labels, not picked off the
    # distribution. Anchoring it on the reference zones instead was circular: they all came
    # from one segment of the scroll that then topped the table.
    print(f"\nfraction of corpus reaching the calibrated legible score:")
    for t in (0.85, 0.875, 0.90, 0.95):
        print(f"  >= {t:.2f}   {int((mh >= t).sum()):>7}   {float((mh >= t).mean()):>7.2%}")

    by_scroll = {}
    for r in stat:
        by_scroll.setdefault(r["scroll"], []).append(r[args.score])
    print(f"\n{'scroll':<14}{'windows':>9}{'mean':>8}{'p90':>8}"
          f"{'legible':>9}{'pct':>8}")
    for s, v in sorted(by_scroll.items(),
                       key=lambda kv: -float((np.array(kv[1]) >= args.legible_at).mean())):
        v = np.array(v)
        leg = int((v >= args.legible_at).sum())
        print(f"{s:<14}{len(v):>9}{v.mean():>8.3f}{np.percentile(v, 90):>8.3f}"
              f"{leg:>9}{leg / len(v):>8.2%}")

    ordered = sorted(rows, key=lambda r: r[args.score])
    tag = f"{args.scroll}_{args.score}" if args.scroll else args.score
    contact_sheet(ordered[-args.n:][::-1], args.index_csv, args.cache_dir,
                  args.cols, args.cell, os.path.join(args.out_dir, f"{tag}_top.png"),
                  args.score)
    contact_sheet(ordered[:args.n], args.index_csv, args.cache_dir,
                  args.cols, args.cell, os.path.join(args.out_dir, f"{tag}_bottom.png"),
                  args.score)
    mid = len(ordered) // 2
    contact_sheet(ordered[mid - args.n // 2:mid + args.n // 2], args.index_csv,
                  args.cache_dir, args.cols, args.cell,
                  os.path.join(args.out_dir, f"{tag}_median.png"), args.score)
    ladder(rows, args.index_csv, args.cache_dir, args.cols, args.cell,
           os.path.join(args.out_dir, f"{tag}_ladder.png"), args.score)


if __name__ == "__main__":
    main()
