"""Turn the atlas from a table into a map: where inside each segment does the pipeline work.

The per-scroll percentages say how much of a scroll is readable but not where, and "where"
is the actionable part - someone attacking a failure needs coordinates, not an average.

Three artefacts:
  maps/<segment>.png   the prediction with the legibility score painted over it
  segment_summary.csv  one row per prediction, ranked by readable fraction
  hotspots.csv         the individual windows worth looking at, with full-res coordinates

Windows overlap (stride is half the window), so scores are accumulated and averaged rather
than written, otherwise the last window silently wins.
"""

import argparse
import csv
import os

import cv2
import numpy as np

DS = 8


def load_atlas(path, score):
    by_pred = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["_s"] = float(r[score])
            r["y0"], r["x0"] = int(r["y0"]), int(r["x0"])
            # Window size is physical, so it differs per prediction; taking it from the row
            # is the only way the painted field lines up with what was actually scored.
            r["_w"] = int(r["window_px"])
            by_pred.setdefault((r["scroll"], r["segment"], r["recipe"]), []).append(r)
    return by_pred


def keys_for(index):
    m = {}
    with open(index, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m.setdefault((r["scroll"], r["segment"], r["recipe"]), r["key"])
    return m


def score_field(rows, shape):
    """Average the overlapping window scores onto a per-pixel field, at ds8 scale."""
    h, w = shape
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for r in rows:
        wpx = r["_w"] // DS
        y, x = r["y0"] // DS, r["x0"] // DS
        acc[y:y + wpx, x:x + wpx] += r["_s"]
        cnt[y:y + wpx, x:x + wpx] += 1
    field = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
    return field, cnt > 0


def render(img, field, covered, thresh, width, lo, hi):
    """Prediction in grey, score as colour, unscored area dimmed.

    Two things had to be tuned or the map is unreadable. The colour range is stretched to
    [lo, hi] rather than [0, 1], because almost every window scores between 0.5 and 0.95 and
    the full range collapses that into one shade. And the tint is kept light: the point is
    to see the score AND the text under it, so a heavy overlay defeats the map.
    """
    h, w = img.shape
    scale = width / w
    size = (width, max(int(h * scale), 1))
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    f = cv2.resize(field, size, interpolation=cv2.INTER_LINEAR)
    c = cv2.resize(covered.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0

    norm = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)
    base = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~c] = (30, 30, 30)

    # Blending the two fought each other: a tint heavy enough to read as a score washes out
    # the very text it is scoring. Two panels keep both legible - prediction above, score
    # below, same geometry so the eye can travel straight down.
    mask = ((f >= thresh) & c).astype(np.uint8)
    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base, cont, -1, (255, 255, 255), 2)
    cv2.drawContours(heat, cont, -1, (255, 255, 255), 2)

    gap = np.full((4, base.shape[1], 3), 20, np.uint8)
    return np.vstack([base, gap, heat])


def colorbar(width, lo, hi, thresh):
    """Without a scale the colours are decorative rather than informative."""
    bar = np.zeros((26, width, 3), np.uint8)
    ramp = np.linspace(0, 255, width).astype(np.uint8)[None, :]
    strip = cv2.applyColorMap(np.repeat(ramp, 12, axis=0), cv2.COLORMAP_TURBO)
    bar[0:12] = strip
    x = int((thresh - lo) / max(hi - lo, 1e-6) * width)
    cv2.line(bar, (x, 0), (x, 11), (255, 255, 255), 2)
    cv2.putText(bar, f"{lo:.2f}", (2, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
    cv2.putText(bar, f"readable {thresh:.3f}", (max(x - 46, 30), 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)
    cv2.putText(bar, f"{hi:.2f}", (width - 30, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (200, 200, 200), 1)
    return bar


def banner(out, text, sub):
    w = out.shape[1]
    bar = np.zeros((44, w, 3), np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(bar, sub, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
    return np.vstack([bar, out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-csv", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--score", default="mass_letter")
    ap.add_argument("--thresh", type=float, default=0.900,
                    help="Calibrated legibility threshold (see calibrate.py)")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--lo", type=float, default=0.30, help="Low end of the colour scale")
    ap.add_argument("--hi", type=float, default=0.95, help="High end of the colour scale")
    ap.add_argument("--maps", type=int, default=40, help="How many segment maps to render")
    ap.add_argument("--min-windows", type=int, default=20,
                    help="Predictions smaller than this are not ranked: a 1-window "
                         "prediction at 100%% readable is noise, not a result")
    ap.add_argument("--hotspots", type=int, default=300)
    args = ap.parse_args()

    map_dir = os.path.join(args.out_dir, "maps")
    os.makedirs(map_dir, exist_ok=True)
    by_pred = load_atlas(args.atlas_csv, args.score)
    key_by_seg = keys_for(args.index_csv)

    summary = []
    for pred, rows in by_pred.items():
        s = np.array([r["_s"] for r in rows])
        best = max(rows, key=lambda r: r["_s"])
        summary.append({
            "scroll": pred[0], "segment": pred[1], "recipe": pred[2],
            "windows": len(rows), "n_text": int((s >= args.thresh).sum()),
            "pct_text": round(float((s >= args.thresh).mean()), 4),
            "mean_score": round(float(s.mean()), 4),
            "max_score": round(float(s.max()), 4),
            "best_y0": best["y0"], "best_x0": best["x0"],
        })
    summary.sort(key=lambda r: (-r["pct_text"], -r["mean_score"]))
    ranked = [r for r in summary if r["windows"] >= args.min_windows]

    out_csv = os.path.join(args.out_dir, "segment_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)
    print(f"wrote {out_csv}  ({len(summary)} predictions)")

    hot = sorted((r for rs in by_pred.values() for r in rs),
                 key=lambda r: -r["_s"])[:args.hotspots]
    hot_csv = os.path.join(args.out_dir, "hotspots.csv")
    with open(hot_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["scroll", "segment", "recipe", "y0", "x0", "window_px", args.score])
        for r in hot:
            wr.writerow([r["scroll"], r["segment"], r["recipe"], r["y0"], r["x0"],
                         r["_w"], f"{r['_s']:.4f}"])
    print(f"wrote {hot_csv}  ({len(hot)} windows)")

    # Render the extremes of the ranking: the segments that worked and the ones that did not.
    half = args.maps // 2
    chosen = ranked[:half] + ranked[-half:]
    for i, srow in enumerate(chosen, 1):
        pred = (srow["scroll"], srow["segment"], srow["recipe"])
        key = key_by_seg.get(pred)
        if not key:
            continue
        img = cv2.imread(os.path.join(args.cache_dir, key.replace("/", "_")),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        field, covered = score_field(by_pred[pred], img.shape)
        out = render(img, field, covered, args.thresh, args.width, args.lo, args.hi)
        out = np.vstack([out, colorbar(out.shape[1], args.lo, args.hi, args.thresh)])
        out = banner(out, f"{srow['scroll']}  {srow['segment']}",
                     f"{srow['pct_text']:.1%} readable  ({srow['n_text']}/{srow['windows']} "
                     f"windows)  mean {srow['mean_score']:.3f}")
        rank = "top" if i <= half else "bot"
        path = os.path.join(map_dir, f"{rank}_{srow['pct_text']:.3f}_{srow['segment'][:24]}.png")
        cv2.imwrite(path, out)
    print(f"wrote {len(chosen)} maps to {map_dir}")

    print(f"\nbest predictions (>= {args.min_windows} windows):")
    print(f"{'scroll':<13}{'segment':<26}{'win':>5}{'text':>6}{'pct':>8}")
    for r in ranked[:12]:
        print(f"{r['scroll']:<13}{r['segment'][:25]:<26}{r['windows']:>5}"
              f"{r['n_text']:>6}{r['pct_text']:>8.1%}")

    dead = [r for r in ranked if r["n_text"] == 0]
    print(f"\npredictions with ZERO readable windows: {len(dead)} of {len(ranked)} ranked "
          f"({len(dead) / len(ranked):.1%})")
    by_scroll = {}
    for r in dead:
        by_scroll[r["scroll"]] = by_scroll.get(r["scroll"], 0) + 1
    for s, n in sorted(by_scroll.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<14}{n:>4}")


if __name__ == "__main__":
    main()
