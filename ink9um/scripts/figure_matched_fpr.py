"""Same false positives, more ink: the soup compared at a matched operating point.

    python scripts/figure_matched_fpr.py <labels_root> <pred_root> <out.png>

A Vesuvius team member objected to the before/after figure that an ensemble can gain "clearness,
like smoothing out the letters" while losing fainter signal. A grey-level picture cannot answer
that, because it has no operating point: any prediction looks like it finds more ink if you are
allowed to move the threshold. This figure removes that freedom.

Each prediction is thresholded at ITS OWN value, chosen so that both make the same false positive
rate on the non-ink pixels of the held-out region: 5%, the level fixed in advance in
`protocols/faint_signal_protocol.md`. The red area is therefore equal between the two panels of a
row BY CONSTRUCTION, and is not a result. What is a result is the green: ink recovered for that
same fixed cost in errors.

Choices that keep it honest, all forced rather than tuned:

1. **All six runs are shown**, three segments times two seeds, including the one run whose overall
   AUC gets worse under the soup. A figure of the best case would be picking the convenient number,
   which is the failure mode this whole experiment exists to check.
2. **Only held-out papyrus**, `validation_mask & ~supervision_mask` at Z=10, never the supervised
   region. Showing the supervised region would display memorisation and call it generalisation.
3. **Every held-out pixel is shown and no window is chosen.** The region arrives as one or two
   connected pieces in opposite corners, so its bounding box can be 92% empty. The pieces are laid
   side by side at the SAME pixel scale, which removes the empty space without removing data, and
   keeps one scale bar valid for the row. The script aborts if the layout loses a single pixel.
4. **No colour map and no saturation limit.** Four discrete outcomes, four flat colours. There is
   no drawing choice left that could manufacture a visible difference, which was the weakness of
   the diverging map in `figure_soup.py`.
5. **The numbers are measured over the whole held-out region before any layout**, so the layout
   cannot move them, and they reproduce the table in `results/faint_signal_results.md`.
"""
import argparse
import os

import numpy as np
import tifffile
import zarr
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, Patch

PITCH_UM = 9.596
PLANE = 10
FPR = 0.05              # fixed in the protocol before any of this was computed
GUTTER = 26             # blank pixels between two pieces of the held-out region

SEGMENTS = ["pherc0814-46527", "pherc0139-w016", "pherc1667-w029"]
SEEDS = (42, 43)

C_TP = "#2e7d32"        # ink found
C_FN = "#9e9e9e"        # ink missed
C_FP = "#d32f2f"        # non-ink called ink
C_TN = "#ffffff"        # non-ink correctly left alone
C_OUT = "#dcdcdc"       # papyrus inside the crop that is not held out
C_PAGE = "#ffffff"      # between the pieces: page, not papyrus


def to_unit(q):
    """Scale by DTYPE, never by the observed maximum, which would depend on the result."""
    if q.dtype == np.uint8:
        return q.astype(np.float32) / 255.0
    if q.dtype == np.uint16:
        return q.astype(np.float32) / 65535.0
    if q.dtype in (np.float32, np.float64):
        return q.astype(np.float32)
    raise SystemExit(f"unsupported dtype {q.dtype}")


def find_pred(out_dir, segment, seed, kind):
    """Same resolution rule as faint_signal.py: naming on disk is not uniform for pherc0814-46527,
    and the soup has float32 and float64 builds; float64 is what produced the published tables."""
    short = "0814" if segment.startswith("pherc0814") else segment
    if kind == "base":
        cands = [f"{segment}_seed{seed}_s75k.tif"]
    else:
        cands = [f"soup_{segment}_A4_seed{seed}_f64.tif",
                 f"soup_{short}_A4_seed{seed}_f64.tif",
                 f"soup_{segment}_A4_seed{seed}.tif",
                 f"soup_{short}_A4_seed{seed}.tif"]
    for c in cands:
        p = os.path.join(out_dir, c)
        if os.path.exists(p):
            return p
    raise SystemExit(f"no {kind} prediction found for {segment} seed{seed}; tried {cands}")


def pieces(held):
    """Bounding box of each connected piece of the held-out region, ordered left to right.

    Nothing is filtered. A dropped piece would mean the figure no longer shows every held-out
    pixel, which is the claim the caption makes, so an unexpected shape fails loudly here.
    """
    lab_, n = ndimage.label(held)
    if n == 0 or n > 4:
        raise SystemExit(f"held-out region has {n} pieces; the side-by-side layout assumes a few")
    boxes = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab_ == k)
        boxes.append((slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)))
    boxes.sort(key=lambda b: b[1].start)
    return boxes


def tile(arrays, boxes):
    """Lay the pieces side by side at the same pixel scale, centred vertically.

    Returns the tiled arrays, a mask of where real data was placed (everything else is page, not
    papyrus), and the rectangle of the leftmost piece so the scale bar can sit on papyrus.
    """
    crops = [[a[b] for b in boxes] for a in arrays]
    shapes = [c.shape for c in crops[0]]
    hgt = max(s[0] for s in shapes)
    wid = sum(s[1] for s in shapes) + GUTTER * (len(boxes) - 1)

    outs = [np.zeros((hgt, wid), dtype=a.dtype) for a in arrays]
    canvas = np.zeros((hgt, wid), dtype=bool)
    x, first = 0, None
    for j, (h, w) in enumerate(shapes):
        y = (hgt - h) // 2
        for o, cl in zip(outs, crops):
            o[y:y + h, x:x + w] = cl[j]
        canvas[y:y + h, x:x + w] = True
        if first is None:
            first = (y, x, h, w)
        x += w + GUTTER
    return outs, canvas, first


def paint(hit, ink, held, canvas):
    """Four outcomes, four flat colours, plus the two kinds of non-papyrus."""
    rgba = np.zeros(hit.shape + (4,), dtype=np.float32)
    rgba[...] = to_rgba(C_TN)
    rgba[hit & ~ink] = to_rgba(C_FP)
    rgba[~hit & ink] = to_rgba(C_FN)
    rgba[hit & ink] = to_rgba(C_TP)
    rgba[~held] = to_rgba(C_OUT)
    rgba[~canvas] = to_rgba(C_PAGE)
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_root")
    ap.add_argument("pred_root")
    ap.add_argument("out")
    ap.add_argument("--plane", type=int, default=PLANE)
    a = ap.parse_args()

    print(f"pitch {PITCH_UM} um/px   plane Z={a.plane}   matched FPR {100 * FPR:.0f}%")
    rows = []
    for seg in SEGMENTS:
        def lab(kind):
            return zarr.open(
                os.path.join(a.labels_root, seg, f"{seg}_{kind}.zarr", "0"), mode="r")

        ink = np.asarray(lab("inklabels")[a.plane]) > 0
        sup = np.asarray(lab("supervision_mask")[a.plane]) > 0
        val = np.asarray(lab("validation_mask")[a.plane]) > 0
        held = val & ~sup

        for seed in SEEDS:
            imgs, stat = {}, {}
            for kind in ("base", "soup"):
                p = find_pred(a.pred_root, seg, seed, kind)
                q = tifffile.imread(p)
                if q.shape != ink.shape:
                    raise SystemExit(f"{seg}/{seed}/{kind}: shapes disagree, that is an "
                                     "alignment bug and not something to plot")
                q = to_unit(q)
                imgs[kind] = q
                # Threshold and recall are computed on the WHOLE held-out region, before any
                # layout, so nothing about the drawing can move the numbers on the figure.
                non, pos = q[held & ~ink], q[held & ink]
                thr = float(np.quantile(non, 1.0 - FPR))
                stat[kind] = dict(thr=thr,
                                  rec=float((pos > thr).mean()),
                                  ach=float((non > thr).mean()),
                                  name=os.path.basename(p))

            boxes = pieces(held)
            (hb, hs, ink_t, held_t), canvas, first = tile(
                [imgs["base"] > stat["base"]["thr"], imgs["soup"] > stat["soup"]["thr"],
                 ink, held], boxes)
            if int(held_t.sum()) != int(held.sum()):
                raise SystemExit("the layout lost held-out pixels, which the caption forbids")

            rows.append(dict(seg=seg, seed=seed, hit={"base": hb, "soup": hs},
                             ink=ink_t, held=held_t, canvas=canvas, first=first,
                             n=int(held.sum()), pieces=len(boxes), stat=stat))
            d = stat["soup"]["rec"] - stat["base"]["rec"]
            print(f"  {seg} seed{seed}   recall {stat['base']['rec']:.4f} -> "
                  f"{stat['soup']['rec']:.4f} ({d:+.4f})   achieved FPR "
                  f"{100 * stat['base']['ach']:.3f}% / {100 * stat['soup']['ach']:.3f}%   "
                  f"{stat['base']['name']} vs {stat['soup']['name']}")

    # Geometry derived from the image shapes rather than chosen: an axes whose aspect is locked by
    # imshow shrinks inside an equal slot and picks up a white band, which reads as if that row
    # had less to show.
    ratios = [r["ink"].shape[0] / r["ink"].shape[1] for r in rows]
    fig_w = 11.5
    lab_in, right_in, gap_h = 2.55, 0.15, 0.20
    head_in, title_in, gap_v, foot_in = 1.22, 0.28, 0.30, 0.78

    col_w = (fig_w - lab_in - right_in - gap_h) / 2.0
    hs = [col_w * r for r in ratios]
    fig_h = head_in + title_in + sum(hs) + gap_v * (len(hs) - 1) + foot_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    axes = np.empty((len(rows), 2), dtype=object)
    for i, h_in in enumerate(hs):
        top = head_in + title_in + sum(hs[:i]) + gap_v * i
        for c in range(2):
            axes[i, c] = fig.add_axes((
                (lab_in + c * (col_w + gap_h)) / fig_w,
                1.0 - (top + h_in) / fig_h,
                col_w / fig_w,
                h_in / fig_h))

    titles = ["released checkpoint, step 75k", "soup of that seed's last four checkpoints"]
    for r, row in enumerate(rows):
        y0, x0, h0, w0 = row["first"]
        for c, kind in enumerate(("base", "soup")):
            ax = axes[r, c]
            ax.imshow(paint(row["hit"][kind], row["ink"], row["held"], row["canvas"]),
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.75")
            if r == 0:
                ax.set_title(titles[c], fontsize=10, pad=8)

        st = row["stat"]
        d = st["soup"]["rec"] - st["base"]["rec"]
        verdict = "ink recovered" if d > 0 else "ink LOST"
        # Horizontal and in a reserved margin: a rotated ylabel is as long as its text, and these
        # rows are only a few centimetres of papyrus tall, so it would run into its neighbours.
        axes[r, 0].text(
            -0.035, 0.5,
            f"{row['seg']}\nseed {row['seed']}\n"
            f"{row['n']:,} held-out px"
            + (f", {row['pieces']} pieces\n" if row["pieces"] > 1 else "\n")
            + f"ink found  {st['base']['rec']:.3f} to {st['soup']['rec']:.3f}\n"
            f"{d:+.3f}  {verdict}\n"
            f"false positives {100 * st['base']['ach']:.2f}% / "
            f"{100 * st['soup']['ach']:.2f}%",
            transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8.5,
            linespacing=1.55)

        bar = 2000.0 / PITCH_UM          # 2 mm, in pixels of this row
        ax = axes[r, 0]
        ax.add_patch(Rectangle((x0 + w0 * 0.05, y0 + h0 * 0.93), bar, max(h0 * 0.014, 3.0),
                               facecolor="black", edgecolor="none"))
        ax.text(x0 + w0 * 0.05, y0 + h0 * 0.915, "2 mm", fontsize=8, va="bottom")

    fig.text(
        0.5, 1.0 - 0.10 / fig_h,
        "Averaging the last four checkpoints of one seed, read at the SAME false positive rate\n"
        "Each panel is thresholded at its own value so that both make false positives on 5% of "
        "the non-ink pixels, so the red area is matched by design and is not a result.\n"
        "The green is: 5 of the 6 runs find more ink for the same errors, and the sixth, which "
        "loses, is shown here too rather than left out.\n"
        "Every pixel shown is held out (validation_mask minus supervision_mask, Z=10): the whole "
        "region, no chosen crop, pieces laid side by side at the same scale.",
        fontsize=9.5, ha="center", va="top")

    fig.legend(handles=[Patch(facecolor=C_TP, label="ink found"),
                        Patch(facecolor=C_FN, label="ink missed"),
                        Patch(facecolor=C_FP, label="false positive (equal area by design)"),
                        Patch(facecolor=C_TN, edgecolor="0.7", label="correctly left alone"),
                        Patch(facecolor=C_OUT, label="not held out")],
               loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.30 / fig_h))
    fig.text(0.5, 0.10 / fig_h,
             "Same inference command and cost as the released checkpoint: what is averaged is the "
             "weights, not the predictions.   Script: ink9um/scripts/figure_matched_fpr.py",
             ha="center", fontsize=8.5, color="0.3")
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
