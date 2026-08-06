"""Draw a stratified, BLIND sample of atlas windows for hand labelling.

The atlas score orders windows well but its absolute cutoff is anchored on two zones from a
single segment of a single scroll - which then tops the resulting table. That is circular.
Breaking it needs labels drawn from across the whole corpus and across the whole score
range, produced without seeing the score.

So the sheets carry an index and nothing else: no score, no scroll, and the windows are
shuffled before rendering. The key linking index to window is written separately and is not
needed until after the labels exist.

Bins are finer at the top because that is where the decision boundary lives; uniform
deciles would spend most of the labelling budget on obvious speckle.

Supplementing an existing label set is supported (`--exclude-key`, `--pitch-in`,
`--start-index`) because a geometry fix moves the window grid and orphans the labels that
were drawn on the old one. Re-labelling only the orphaned arms keeps the rest of the budget
intact, and the arms that a fix touched are precisely the ones that must not go unverified.
"""

import argparse
import csv
import os

import cv2
import numpy as np

DS = 8

BINS = [(0.00, 0.30), (0.30, 0.50), (0.50, 0.65), (0.65, 0.75), (0.75, 0.82),
        (0.82, 0.87), (0.87, 0.90), (0.90, 0.93), (0.93, 0.95), (0.95, 1.01)]


def load_atlas(path, score):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_s"] = float(r[score])
        r["y0"], r["x0"] = int(r["y0"]), int(r["x0"])
    return rows


def keys_for(index):
    m = {}
    with open(index, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m.setdefault((r["scroll"], r["segment"], r["recipe"]), r["key"])
    return m


def stratified(rows, per_bin, rng):
    """Spread each bin across scrolls so one dominant scroll cannot own the calibration.

    PHercParis4 alone is 90% of the corpus, so a plain random draw inside a bin would be a
    PHercParis4 draw. Round-robin over scrolls until the bin quota is filled.
    """
    picked = []
    for lo, hi in BINS:
        pool = [r for r in rows if lo <= r["_s"] < hi]
        if not pool:
            continue
        by_scroll = {}
        for r in pool:
            by_scroll.setdefault(r["scroll"], []).append(r)
        for v in by_scroll.values():
            rng.shuffle(v)
        order = sorted(by_scroll, key=lambda s: -len(by_scroll[s]))
        take, i = [], 0
        while len(take) < min(per_bin, len(pool)):
            s = order[i % len(order)]
            if by_scroll[s]:
                take.append(by_scroll[s].pop())
            i += 1
            if all(not by_scroll[s] for s in order):
                break
        for r in take:
            r["_bin"] = f"{lo:.2f}-{hi:.2f}"
        picked += take
    return picked


def render(sample, index, cache_dir, out_dir, per_sheet, cols, cell):
    key_by_seg = keys_for(index)
    cache = {}

    sheet, tiles = 0, []
    for r in sample:
        key = key_by_seg.get((r["scroll"], r["segment"], r["recipe"]))
        path = os.path.join(cache_dir, key.replace("/", "_"))
        if path not in cache:
            cache.clear()  # windows are grouped by nothing; keep memory flat
            cache[path] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cache[path]
        if img is None:
            continue
        # The window is a physical size, so its pixel size varies per prediction.
        wpx = int(r["window_px"]) // DS
        y, x = r["y0"] // DS, r["x0"] // DS
        win = img[y:y + wpx, x:x + wpx]
        if win.size == 0:
            continue
        win = cv2.resize(win, (cell, cell), interpolation=cv2.INTER_AREA)
        win = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(win, (0, 0), (cell - 1, 20), (0, 0, 0), -1)
        # Index only. No score, no scroll: the label must not be primed by the number
        # it is meant to validate.
        cv2.putText(win, f"#{r['_i']:03d}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 220, 255), 1)
        cv2.rectangle(win, (0, 0), (cell - 1, cell - 1), (70, 70, 70), 1)
        tiles.append(win)

        if len(tiles) == per_sheet:
            write_sheet(tiles, cols, cell, out_dir, sheet)
            sheet, tiles = sheet + 1, []
    if tiles:
        write_sheet(tiles, cols, cell, out_dir, sheet)
        sheet += 1
    return sheet


def write_sheet(tiles, cols, cell, out_dir, n):
    rows_n = (len(tiles) + cols - 1) // cols
    pad = list(tiles)
    while len(pad) < rows_n * cols:
        pad.append(np.zeros((cell, cell, 3), np.uint8))
    grid = np.vstack([np.hstack(pad[i * cols:(i + 1) * cols]) for i in range(rows_n)])
    out = os.path.join(out_dir, f"sheet_{n:02d}.png")
    cv2.imwrite(out, grid)
    print(f"wrote {out}  ({len(tiles)} windows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-csv", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--score", default="mass_letter")
    ap.add_argument("--pitch-in", default="",
                    help="Comma list of pitch_um values to draw from; default all")
    ap.add_argument("--exclude-key", default="",
                    help="Existing labeling key whose windows must not be drawn again")
    ap.add_argument("--start-index", type=int, default=1,
                    help="First index to hand out, so a supplement does not collide "
                         "with the labels it is supplementing")
    ap.add_argument("--per-bin", type=int, default=12)
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = load_atlas(args.atlas_csv, args.score)

    if args.pitch_in:
        want = {p.strip() for p in args.pitch_in.split(",")}
        rows = [r for r in rows if r["pitch_um"] in want]
        print(f"restricted to pitch {sorted(want)}: {len(rows)} windows")
    if args.exclude_key:
        with open(args.exclude_key, encoding="utf-8") as f:
            seen = {(r["scroll"], r["segment"], r["recipe"], r["y0"], r["x0"])
                    for r in csv.DictReader(f)}
        rows = [r for r in rows
                if (r["scroll"], r["segment"], r["recipe"], str(r["y0"]), str(r["x0"]))
                not in seen]
        print(f"after excluding {len(seen)} already-drawn windows: {len(rows)}")

    sample = stratified(rows, args.per_bin, rng)
    rng.shuffle(sample)  # sheet order must carry no information about the score
    for i, r in enumerate(sample, args.start_index):
        r["_i"] = i

    key_path = os.path.join(args.out_dir, "labeling_key.csv")
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["i", "scroll", "segment", "recipe", "y0", "x0", "bin", args.score])
        for r in sample:
            wr.writerow([r["_i"], r["scroll"], r["segment"], r["recipe"], r["y0"],
                         r["x0"], r["_bin"], f"{r['_s']:.4f}"])

    n = render(sample, args.index_csv, args.cache_dir, args.out_dir,
               args.per_sheet, args.cols, args.cell)

    todo = os.path.join(args.out_dir, "labels.csv")
    if not os.path.exists(todo):
        with open(todo, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["i", "label"])
    print(f"\n{len(sample)} windows in {n} sheets")
    print(f"key : {key_path}")
    print(f"fill: {todo}   labels = text | partial | speckle")

    by_bin = {}
    for r in sample:
        by_bin.setdefault(r["_bin"], []).append(r["scroll"])
    print(f"\n{'bin':<12}{'n':>4}  scrolls")
    for b, v in sorted(by_bin.items()):
        u = sorted({s: v.count(s) for s in set(v)}.items(), key=lambda kv: -kv[1])
        print(f"{b:<12}{len(v):>4}  " + " ".join(f"{s}:{c}" for s, c in u))


if __name__ == "__main__":
    main()
