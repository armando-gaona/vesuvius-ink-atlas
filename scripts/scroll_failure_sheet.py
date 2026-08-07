"""Look at the concentrated failure before claiming it is one.

57 of the 70 predictions with nothing over the threshold anywhere on their raster are a
single scroll under a single recipe. That is either the most actionable line in the atlas or
the fourth time this scroll has caught a bug of mine, and prose cannot tell the difference.

So this renders two sheets that answer it by eye:

  failures  - the BEST window of each failing prediction, the most legible thing it offers.
              If these are speckle, the prediction really does produce nothing.
  control   - the top-scoring windows of the SAME scroll that do clear the threshold.
              If these are readable Greek, the metric works here and the failure is real.
              If they are also speckle, the metric is broken on this scroll and the whole
              finding collapses.

Without the control the failure sheet proves nothing: a metric that scores everything low on
a scroll produces exactly the same picture whether the pipeline failed or the metric did.
"""

import argparse
import csv
import os

import cv2
import numpy as np

DS = 8


def crop(cache_dir, key, y0, x0, window_px, cell):
    local = os.path.join(cache_dir, key.replace("/", "_"))
    img = cv2.imread(local, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Windows are a fixed physical size, so their pixel size follows the pitch the prediction
    # was rendered at. Cropping a constant number of pixels would show a different region
    # than the one that was scored - the mistake that cost this project three headlines.
    wpx = int(window_px) // DS
    y, x = int(y0) // DS, int(x0) // DS
    win = img[y:y + wpx, x:x + wpx]
    if win.size == 0:
        return None
    # Windows that run off the edge of the raster come back short. Resizing those straight to
    # a square stretches them, and a stretched letter is not a letter you can judge - the eye
    # reads squashed speckle as text and squashed text as speckle. Pad to square instead, so
    # every cell shows the same physical shape and the black margin says "edge of segment".
    if win.shape[0] != wpx or win.shape[1] != wpx:
        pad = np.zeros((wpx, wpx), win.dtype)
        pad[:win.shape[0], :win.shape[1]] = win
        win = pad
    return cv2.resize(win, (cell, cell), interpolation=cv2.INTER_AREA)


def tile(win, cell, label, bar=None):
    win = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
    h = max(18, cell // 18)
    cv2.rectangle(win, (0, 0), (cell - 1, h), (0, 0, 0), -1)
    cv2.putText(win, label, (4, h - 5), cv2.FONT_HERSHEY_SIMPLEX,
                h / 47.0, (0, 220, 255), 1)
    # A scale bar is the difference between "these look like letters" and "these ARE letters".
    # Without it every cell is the same square and the eye has no idea whether the marks in it
    # are 1 mm or 1 cm across - which is exactly the confusion that has already cost this
    # project three retracted headlines.
    if bar:
        px, txt = bar
        y = cell - 14
        cv2.rectangle(win, (8, y - 4), (8 + px, y + 2), (0, 0, 0), -1)
        cv2.rectangle(win, (8, y - 4), (8 + px, y + 2), (0, 220, 255), -1)
        cv2.putText(win, txt, (8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
    cv2.rectangle(win, (0, 0), (cell - 1, cell - 1), (60, 60, 60), 1)
    return win


def sheet(tiles, cols, cell, out_path, per_sheet=0):
    """Write the tiles as one sheet, or as numbered pages of `per_sheet` tiles each.

    Pagination exists for one reason: a sheet is only evidence if it can be read. A 70-cell
    sheet gets downscaled to fit whatever is looking at it, and at that size a letter and a
    speck of noise are the same grey smudge. Small pages keep every cell at full size, which
    is the whole point of rendering them.
    """
    if not tiles:
        print(f"no tiles for {out_path}")
        return
    if not per_sheet:
        pages = [tiles]
    else:
        pages = [tiles[i:i + per_sheet] for i in range(0, len(tiles), per_sheet)]
    for p, page in enumerate(pages):
        page = list(page)
        n = (len(page) + cols - 1) // cols
        while len(page) < n * cols:
            page.append(np.zeros((cell, cell, 3), np.uint8))
        img = np.vstack([np.hstack(page[i * cols:(i + 1) * cols]) for i in range(n)])
        path = out_path if len(pages) == 1 else out_path.replace(".png", f"_p{p:02d}.png")
        cv2.imwrite(path, img)
        print(f"wrote {path}  ({n}x{cols} cells)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures-csv", default="data/failures.csv")
    ap.add_argument("--atlas-csv", default="data/atlas.csv")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--scroll", default="PHerc0172")
    ap.add_argument("--thresh", type=float, default=0.900)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--control-n", type=int, default=24)
    ap.add_argument("--per-sheet", type=int, default=0,
                    help="Split the failure sheet into pages of N cells (0 = one sheet)")
    ap.add_argument("--window-mm", type=float, default=19.652,
                    help="Physical side of an atlas window, for the scale bar")
    args = ap.parse_args()

    with open(args.failures_csv, encoding="utf-8") as f:
        fail = [r for r in csv.DictReader(f)
                if int(r["n_text_all"]) == 0
                and (args.scroll in ("ALL", "") or r["scroll"] == args.scroll)]
    print(f"{len(fail)} clean failures in {args.scroll}")

    # Every window covers the same physical area, so one bar length serves every cell.
    bar = (int(round(args.cell * 5.0 / args.window_mm)), "5mm")

    tiles, missing = [], 0
    for i, r in enumerate(sorted(fail, key=lambda r: -float(r["max_score"]))):
        win = crop(args.cache_dir, r["key"], r["best_y0"], r["best_x0"], r["window_px"],
                   args.cell)
        if win is None:
            missing += 1
            continue
        print(f"  #{i:02d}  {float(r['max_score']):.3f}  {r['scroll']:<12} "
              f"{r['segment']:<16} {r['recipe']}")
        tiles.append(tile(win, args.cell,
                          f"#{i:02d} {float(r['max_score']):.2f} {r['scroll'][:9]}", bar))
    sheet(tiles, args.cols, args.cell,
          os.path.join(args.out_dir, f"{args.scroll}_failures_best.png"), args.per_sheet)
    if missing:
        print(f"  {missing} predictions had no cached raster")

    with open(args.atlas_csv, encoding="utf-8") as f:
        pas = [r for r in csv.DictReader(f)
               if (args.scroll in ("ALL", "") or r["scroll"] == args.scroll)
               and float(r["mass_letter"]) >= args.thresh]
    print(f"{len(pas)} windows in {args.scroll} clear {args.thresh} on the full grid")

    tiles = []
    for r in sorted(pas, key=lambda r: -float(r["mass_letter"]))[:args.control_n]:
        win = crop(args.cache_dir, r["key"], r["y0"], r["x0"], r["window_px"], args.cell)
        if win is None:
            continue
        tiles.append(tile(win, args.cell,
                          f"{float(r['mass_letter']):.2f} {r['scroll'][:9]}", bar))
    sheet(tiles, args.cols, args.cell,
          os.path.join(args.out_dir, f"{args.scroll}_passing_top.png"), args.per_sheet)


if __name__ == "__main__":
    main()
