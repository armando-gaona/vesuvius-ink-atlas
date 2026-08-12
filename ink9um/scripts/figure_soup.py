"""Before/after figure for the weight soup, on held-out papyrus only.

    python scripts/figure_soup.py <labels_root> <pred_root> <out.png>

Three rows, one per segment that ships a `_validation_mask.zarr`. Three columns: the released
seed-42 checkpoint at step 75k, the soup of that seed's last four checkpoints, and the
difference between them.

Choices that keep the picture honest, all of them forced rather than tuned:

1. **Only the held-out region is shown**, `validation_mask & ~supervision_mask` at Z=10, which
   is papyrus the model never saw. Showing the supervised region would display memorisation and
   call it generalisation.
2. **Every held-out pixel is shown, and no window is chosen.** The held-out region comes in one
   or two connected pieces sitting at opposite corners of the segment, so its bounding box is up
   to 92% empty. The pieces are therefore laid side by side at the SAME pixel scale, in a fixed
   left-to-right order, which removes the empty space without removing any data and without
   anyone picking a crop. Scale is preserved, so one scale bar per row is still correct. The
   rows have very different shapes and are sized accordingly, so each carries its own bar.
3. **Grey levels are absolute**, 0 to 1 on both prediction columns. Auto-scaling each panel to
   its own range would manufacture a visible difference out of nothing.
4. **The difference column carries the annotation outline.** The claim is not "it got darker",
   it is "it got darker on the strokes and not off them", and only ground truth drawn on top
   makes that checkable.
5. **The AUC on each row is measured over exactly the pixels displayed**, so the number and the
   picture describe the same papyrus. It is measured over the whole held-out region of the
   segment, before the pieces are laid out, so the layout cannot move it.
"""
import argparse
import os

import numpy as np
import tifffile
import zarr
from scipy import ndimage
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

PITCH_UM = 9.596
DIFF_LIMIT = 0.30       # saturation of the difference colour map
OUTSIDE = "#dcdcdc"     # papyrus that is not held out
GUTTER = 26             # blank pixels between two pieces of the held-out region

# Seed 42 is the one DISPLAYED, because a picture can only show one run. The published
# headline is the mean over the two seeds the project released, so both seeds are scored here
# and both numbers are printed on the figure. Showing one seed and labelling it with the
# two-seed mean would be a quiet mismatch between the picture and the caption.
SEGMENTS = [
    ("pherc0814-46527",
     "pherc0814-46527_seed42_s75k.tif", "soup_0814_A4_seed42.tif",
     "pherc0814-46527_seed43_s75k.tif", "soup_0814_A4_seed43.tif"),
    ("pherc0139-w016",
     "pherc0139-w016_seed42_s75k.tif", "soup_pherc0139-w016_A4_seed42.tif",
     "pherc0139-w016_seed43_s75k.tif", "soup_pherc0139-w016_A4_seed43.tif"),
    ("pherc1667-w029",
     "pherc1667-w029_seed42_s75k.tif", "soup_pherc1667-w029_A4_seed42.tif",
     "pherc1667-w029_seed43_s75k.tif", "soup_pherc1667-w029_A4_seed43.tif"),
]


def to_unit(q):
    """Scale by DTYPE, never by the observed maximum, which would depend on the result."""
    if q.dtype == np.uint8:
        return q.astype(np.float32) / 255.0
    if q.dtype == np.uint16:
        return q.astype(np.float32) / 65535.0
    if q.dtype in (np.float32, np.float64):
        return q.astype(np.float32)
    raise SystemExit(f"unsupported dtype {q.dtype}")


def auc(scores, labels):
    r = rankdata(scores)
    npos = int(labels.sum())
    nneg = labels.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def outline(mask):
    """Thin inner boundary of a binary mask, so ground truth never hides the image."""
    e = np.zeros_like(mask)
    e[:-1, :] |= mask[:-1, :] ^ mask[1:, :]
    e[:, :-1] |= mask[:, :-1] ^ mask[:, 1:]
    return e & mask


def pieces(held):
    """Bounding box of each connected piece of the held-out region, ordered left to right.

    Nothing is filtered: a dropped piece would mean the figure no longer shows every held-out
    pixel, which is the whole claim of the panel. If the mask ever came back as dust the layout
    would be meaningless, so that fails loudly here instead of rendering something unreadable.
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

    Returns the tiled arrays, a mask of where real data was placed (everything else is page,
    not papyrus), and the rectangle of the leftmost piece so the scale bar can sit on it.
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


def paint(img, shown, canvas, cmap, vmin, vmax):
    """Explicit RGBA, because three different things need three different colours: the value,
    the papyrus that is not held out, and the page between the pieces."""
    rgba = cmap(np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0))
    rgba[~shown] = to_rgba(OUTSIDE)
    rgba[~canvas] = (1.0, 1.0, 1.0, 1.0)
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_root")
    ap.add_argument("pred_root")
    ap.add_argument("out")
    ap.add_argument("--plane", type=int, default=10)
    a = ap.parse_args()

    rows = []
    for seg, base42, soup42, base43, soup43 in SEGMENTS:
        imgs = {k: to_unit(tifffile.imread(os.path.join(a.pred_root, v)))
                for k, v in (("b42", base42), ("s42", soup42),
                             ("b43", base43), ("s43", soup43))}

        def lab(kind):
            return zarr.open(
                os.path.join(a.labels_root, seg, f"{seg}_{kind}.zarr", "0"), mode="r")

        ink = np.asarray(lab("inklabels")[a.plane]) > 0
        sup = np.asarray(lab("supervision_mask")[a.plane]) > 0
        val = np.asarray(lab("validation_mask")[a.plane]) > 0
        for k, v in imgs.items():
            if v.shape != ink.shape:
                raise SystemExit(
                    f"{seg}/{k}: shapes disagree, comparing them would be an alignment bug")
        held = val & ~sup

        # Scored on the whole held-out region, before any layout, so the layout cannot move it.
        sc = {k: auc(v[held], ink[held]) for k, v in imgs.items()}
        mean_base = 0.5 * (sc["b42"] + sc["b43"])
        mean_soup = 0.5 * (sc["s42"] + sc["s43"])

        boxes = pieces(held)
        (base, soup, ink_t, held_t), canvas, first = tile(
            [imgs["b42"], imgs["s42"], ink, held], boxes)
        if int(held_t.sum()) != int(held.sum()):
            raise SystemExit("the layout lost held-out pixels, which the caption forbids")

        rows.append(dict(seg=seg, base=base, soup=soup, ink=ink_t, held=held_t,
                         canvas=canvas, first=first, n=int(held.sum()), pieces=len(boxes),
                         a_base=sc["b42"], a_soup=sc["s42"],
                         m_base=mean_base, m_soup=mean_soup))
        h, w = base.shape
        print(f"{seg}: seed42 {sc['b42']:.4f} -> {sc['s42']:.4f}   "
              f"seed43 {sc['b43']:.4f} -> {sc['s43']:.4f}   "
              f"two-seed mean {mean_base:.4f} -> {mean_soup:.4f} ({mean_soup - mean_base:+.4f})   "
              f"{int(held.sum()):,} px in {len(boxes)} piece(s), panel "
              f"{h * PITCH_UM / 1000:.1f} x {w * PITCH_UM / 1000:.1f} mm")

    # Each row keeps its own aspect, so the figure height is DERIVED from the shapes rather than
    # chosen. Picking it by eye leaves a row floating in whitespace, which reads as if that row
    # had less to show.
    ratios = [r["base"].shape[0] / r["base"].shape[1] for r in rows]
    # The axes are placed by hand, in inches. `subplots` plus `tight_layout` hands every row an
    # equal-ish slot, and an axes whose aspect is locked by imshow then shrinks inside its slot
    # and is centred there, so each row picks up a white band whose size depends on rounding.
    # Deriving the geometry from the image shapes makes each panel exactly its own aspect.
    fig_w = 12.0
    lab_in, right_in, gap_h = 2.30, 0.15, 0.18    # lab_in holds the per-row numbers
    head_in, title_in, gap_v, foot_in = 0.72, 0.30, 0.24, 0.38

    col_w = (fig_w - lab_in - right_in - 2 * gap_h) / 3.0
    hs = [col_w * r for r in ratios]
    fig_h = head_in + title_in + sum(hs) + gap_v * (len(hs) - 1) + foot_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = np.empty((3, 3), dtype=object)
    for i, h_in in enumerate(hs):
        top = head_in + title_in + sum(hs[:i]) + gap_v * i
        for c in range(3):
            axes[i, c] = fig.add_axes((
                (lab_in + c * (col_w + gap_h)) / fig_w,
                1.0 - (top + h_in) / fig_h,
                col_w / fig_w,
                h_in / fig_h))
    fig.patch.set_facecolor("white")
    titles = ["released checkpoint (seed 42, step 75k)",
              "soup of that seed's last four",
              "soup minus released"]

    grey = plt.get_cmap("gray_r")
    diverging = plt.get_cmap("bwr")

    for r, row in enumerate(rows):
        edge = outline(row["ink"] & row["held"])
        y0, x0, h0, w0 = row["first"]
        for c in range(3):
            if c < 2:
                img = paint((row["base"], row["soup"])[c], row["held"], row["canvas"],
                            grey, 0.0, 1.0)
            else:
                img = paint(row["soup"] - row["base"], row["held"], row["canvas"],
                            diverging, -DIFF_LIMIT, DIFF_LIMIT)
                img[edge] = (0.0, 0.0, 0.0, 1.0)
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.75")
            if r == 0:
                ax.set_title(titles[c], fontsize=10, pad=8)

        # Horizontal, in a reserved margin. A rotated ylabel is as long as its text, and these
        # rows are only a few centimetres of papyrus tall, so it would run into its neighbours.
        axes[r, 0].text(
            -0.035, 0.5,
            f"{row['seg']}\n"
            f"{row['n']:,} held-out px"
            + (f", {row['pieces']} pieces\n" if row["pieces"] > 1 else "\n")
            + f"AUC, seed 42 shown here\n"
            f"   {row['a_base']:.4f} to {row['a_soup']:.4f} "
            f"({row['a_soup'] - row['a_base']:+.4f})\n"
            f"AUC, mean of both seeds\n"
            f"   {row['m_base']:.4f} to {row['m_soup']:.4f} "
            f"({row['m_soup'] - row['m_base']:+.4f})",
            transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8.5,
            linespacing=1.5)

        bar = 2000.0 / PITCH_UM          # 2 mm, in pixels of this row
        ax = axes[r, 0]
        ax.add_patch(Rectangle((x0 + w0 * 0.05, y0 + h0 * 0.94), bar, max(h0 * 0.012, 3.0),
                               facecolor="black", edgecolor="none"))
        ax.text(x0 + w0 * 0.05, y0 + h0 * 0.925, "2 mm", fontsize=8.5, va="bottom")

    fig.text(
        0.5, 1.0 - 0.08 / fig_h,
        "Averaging the last four checkpoints of one seed, on papyrus the model never saw\n"
        "Every pixel shown is held out (validation_mask minus supervision_mask, Z=10): the "
        "whole region, no chosen crop, two pieces laid side by side at the same scale.\n"
        "Grey is absolute 0 to 1. Third column: red is where the soup reads more ink, blue "
        "less, black outline is the annotation. Images are seed 42.",
        fontsize=10, ha="center", va="top")
    fig.text(0.5, 0.12 / fig_h,
             "Same inference command and cost as the released checkpoint: what is averaged is "
             "the weights, not the predictions.   Script: ink9um/scripts/figure_soup.py",
             ha="center", fontsize=8.5, color="0.3")
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
