# Ink Prediction Failure Atlas

**A calibrated, objective legibility metric applied to every published ink prediction in the
Vesuvius Challenge open bucket — and the resulting map of where the pipeline works and where
it does not.**

Today there is no objective criterion for "is this prediction readable?". The de-facto
standard is *a person looked at it*. This repository turns that into a number, calibrates
the number against blind hand labels, and applies it to the whole published corpus.

It needs **no GPU and no model**: it scores predictions that the project has already
published, so the cost is bandwidth.

> Motivated by the project's own framing in
> [2026 Open Problems](https://scrollprize.org/2026_open_problems):
> *"This is why better diagnostics matter just as much as better models — without them,
> it's hard to know which of these failure modes you're even fighting."*

---

## Headline results

| | |
|---|---|
| Predictions indexed | **423** (226 segments, 7 scrolls) |
| Windows scored | **46,764**, each covering an identical **19.65 × 19.65 mm** of papyrus |
| Legibility metric | `mass_letter` — ink mass in connected components 960–7200 µm across |
| Validation | **AUC 0.930** against 125 blind, stratified hand labels |
| Calibrated threshold | **0.900** (best F1 = 0.804; precision 0.804, recall 0.804) |
| **Legible fraction of the corpus** | **4.08%** (1,910 of 46,764 windows) |
| **Predictions with zero readable windows** | **82 of 209 ranked (39.2%)** |

Because every window covers the same *physical* area, "% of windows" is literally
**"% of papyrus area"**.

### Per scroll, at the calibrated threshold

| scroll | windows | legible | % |
|---|---:|---:|---:|
| PHerc1667 | 1,237 | 98 | **7.92%** |
| PHercParis4 | 31,605 | 1,731 | 5.48% |
| PHerc0814 | 94 | 1 | 1.06% |
| PHerc0172 | 12,344 | 74 | 0.60% |
| PHerc0139 | 1,407 | 6 | 0.43% |
| PHerc0343P | 12 | 0 | 0.00% |
| PHerc0500P2 | 65 | 0 | 0.00% |

Predictions with **zero** readable windows, by scroll: PHerc0139 31, PHerc0172 27,
PHercParis4 16, PHerc1667 8. That list is the actionable output — see
[`data/segment_summary.csv`](data/segment_summary.csv).

---

## ⚠️ A data bug that affects anyone working at physical scale

**`resolution_um` in a published prediction's filename names the SOURCE SCAN, not the pixel
pitch of the raster.** Renders are frequently taken from a downsampled level of the
multiscale pyramid.

| recipe | µm in the filename | `source_group` | **actual raster pitch** |
|---|---|---|---|
| canon | 2.399 | 0 | 2.399 µm/px |
| `1um_s1z2` | 1.129 | **1** | **2.258 µm/px** |

The `1um_s1z2` predictions are rendered from **level 1** of the pyramid, so their true pitch
is **2× coarser** than their name declares. This affects **112 of 423** published predictions.

**How this was caught, without trusting any metadata** ([`scripts/check_pitch.py`](scripts/check_pitch.py)):
113 segments are published under two recipes. Those are the same physical surface, so
`width_px × pitch_um` must match. Result: **113/113 have the same pixel dimensions**
(ratio 1.06) but physical widths differing by **×2.00**. Impossible for the same papyrus.
Confirmed authoritatively by reading `source_group` from each surface-volume zarr's
`.zattrs` ([`scripts/fetch_pitch.py`](scripts/fetch_pitch.py)).

**What it cost us.** An earlier version of this analysis found recipe `canon` beating
`1um_s1z2` in **36 of 36** paired segments (sign test p ≈ 2.9e-11). After the fix: **14/36,
p = 0.243**. The sweep was *entirely* the bug. **That finding is retracted — there is no
detectable difference between the two recipes.**

If you build anything that compares sizes, resolutions, or areas across this corpus, read
the pitch from the zarr, not from the filename.

---

## Method, and why each piece is the way it is

### Legibility is a property of *scale*, not of intensity

Intensity signals (99th-percentile logit, positive fraction, probability std) order known
readable and unreadable zones correctly but with margins too small to threshold. Spatial
structure separates them cleanly: a letter is ~3 mm across, so text puts nearly all its ink
mass into large connected components, and a speckle field puts none.

### The metric is a *band*, not a lower bound

A saturated prediction is **one enormous connected component** and scores just as well as a
page of letters under a lower-bound metric. Half of an early top-24 was white smear.
`mass_letter` requires components to be **between** 960 µm and 7200 µm — failing off both
ends, with a distinct failure mode reported on each side (`mass_smear` above the band,
speckle below).

Sanity check that it is not a brightness proxy: `corr(mass_letter, frac_above) = 0.203`.

### Everything comparable across scans is measured in microns

This is the single most important rule in the repository, and it was learned by getting it
wrong three separate times:

1. **Component size in pixels** → PHerc0172 (the only 7.91 µm scan) was asked for letters 3×
   larger than it can physically have. It ranked last in the corpus while its contact sheets
   showed some of the cleanest Greek columns available.
2. **`resolution_um` read as the pitch** → the retracted 36/36 recipe result above.
3. **Analysis window in fixed pixels** → 8192 px covers 19.7 mm at 2.4 µm/px but 64.8 mm at
   7.91 µm/px, **11× the physical area**. A mass fraction over 11× the area dilutes, so a
   fixed-pixel window penalises the coarsest scan by construction. Telltale: PHerc0172
   averaged **1,659 connected components per window** against 90–140 everywhere else. After
   the fix: 166. Its legible fraction went from 0.00% to 0.60%.

All three produced results that *looked like findings*, and all three collapsed on
inspection of the pixels. If you see a threshold in pixels in this codebase, treat it as a
bug until proven otherwise.

### Per-image Otsu, not a fixed threshold

Recipes differ in contrast — PHerc0172's predictions are faint continuous tone, PHercParis4's
are near-binary white. At a fixed cutoff the metric measures prediction contrast rather than
legibility. Otsu is computed **over on-segment pixels only**: off-segment area is exact zero
and drags the split down.

### Calibration is blind and stratified

The first threshold was anchored on 2 hand-verified zones from **one segment of one scroll**
— which then topped the resulting table. Circular. It was broken by blind labelling:

- 125 windows, stratified over 10 score bands (finer at the top, where the decision boundary
  lives), and **round-robin across scrolls within each band** — PHercParis4 is 68% of the
  corpus, so a plain draw inside a band is a PHercParis4 draw.
- Sheets rendered with **an index and nothing else**: no score, no scroll name, shuffled
  order. Labelling while looking at the number you intend to validate manufactures the
  answer. The index→window key is written separately
  ([`data/labels/labeling_key.csv`](data/labels/labeling_key.csv)).
- Labels: `text` / `partial` / `speckle` → 46 / 38 / 41.

| label | n | mean | p25 | median | p75 |
|---|---:|---:|---:|---:|---:|
| text | 46 | **0.928** | 0.906 | 0.938 | 0.954 |
| partial | 38 | 0.788 | 0.758 | 0.830 | 0.891 |
| speckle | 41 | **0.574** | 0.414 | 0.588 | 0.752 |

**AUC = 0.930** (threshold-free, so it cannot be tuned to look good). Full sweep in
[`results/calibration.txt`](results/calibration.txt).

![corpus distribution and the blind labels against the score](figures/calibration.png)

The right-hand panel is the honest picture. `text` and `speckle` separate cleanly; `partial`
straddles the line and some speckle reaches 0.75, which is why precision at the calibrated
threshold is 0.80 and not higher.

---

## Honest limitations

Read these before citing any number above.

- **Precision 0.80 at the chosen threshold: ~2 in 10 flagged windows are not text.** This is
  a *screener*, not a verdict. It tells you where to look, not what is there.
- **The 125 labels were produced by an AI assistant under human supervision, not by a
  papyrologist.** They are blind, stratified and fully reproducible (sheets, key and labels
  are all in this repo), but they are **not expert ground truth**. Independent re-labelling
  is the single highest-value contribution anyone could make to this work — the protocol is
  in [`scripts/sample_for_labeling.py`](scripts/sample_for_labeling.py) and takes about an
  hour.
- **PHercParis4 is 68% of all windows.** Per-scroll figures are comparable to each other;
  global averages are de facto PHercParis4 averages and must not be cited without this note.
- **The metric is a reliable detector of extremes, not a graded measure of legibility.** The
  top and bottom of the ranking are verified in both directions (see
  [`figures/mass_letter_top.png`](figures/mass_letter_top.png) — uniformly clean Greek — and
  [`figures/mass_letter_bottom.png`](figures/mass_letter_bottom.png) — uniformly sparse
  specks). In the bulk of the distribution it is **not visually monotonic**; see the decile
  ladder, [`figures/mass_letter_ladder.png`](figures/mass_letter_ladder.png).
- **`speckle` reaches 0.752 at its p75** — the metric over-scores dense speckle.
- **This measures published *predictions*, not scroll content.** A region scoring 0 may hold
  perfectly good text that the current pipeline failed to recover. That is the point: this
  is a failure atlas, not a census of ink.
- **`data/labels/labels.csv` holds 160 label rows, of which 125 join** to the current window
  grid. The geometry fix (item 3 above) moved the grid and orphaned 35 earlier labels;
  they are kept for traceability, and the 40 replacements were drawn blind over exactly the
  arms the fix touched (`--pitch-in 2.258,7.91,2.215`), so the correction did not go
  unverified.

---

## What is in here

```
data/
  atlas.csv                46,764 scored windows — the primary artifact
  segment_summary.csv      one row per prediction, ranked by readable fraction
  hotspots.csv             top 300 windows with full-resolution coordinates
  predictions_index.csv    the 423 published predictions found in the bucket
  predictions_pitch.csv    each one's TRUE raster pitch, read from .zattrs (421/423)
  labels/
    labeling_key.csv       index -> window, written before any label existed
    labels.csv             the blind labels
figures/
  mass_letter_top.png      top-scoring windows, score burned in
  mass_letter_bottom.png   bottom-scoring windows
  mass_letter_median.png   the middle of the distribution
  mass_letter_ladder.png   one row per score decile — the calibration artifact
  maps/                    per-segment maps: prediction above, score field below
results/
  calibration.txt          full threshold sweep and per-scroll breakdown
notebooks/
  atlas_walkthrough.ipynb  reproduce the headline numbers from data/atlas.csv
scripts/                   see below
```

### `data/atlas.csv` columns

| column | meaning |
|---|---|
| `scroll`, `segment`, `recipe` | which published prediction |
| `resolution_um` | the value in the filename — **the scan, not the pitch** |
| `pitch_um` | the true raster pitch, from the zarr `.zattrs` |
| `window_px` | window side in full-resolution px (varies with pitch; physical size is constant) |
| `thresh` | the per-image Otsu cutoff used |
| `y0`, `x0` | window origin in **full-resolution** coordinates |
| `coverage` | fraction of the window that is on-segment |
| `frac_above` | fraction of pixels above threshold — "amount of ink" |
| `n_comp` | connected components in the window |
| `median_extent` | median component extent, µm |
| `mass_big`, `mass_huge` | mass in components ≥960 µm / ≥1920 µm |
| **`mass_letter`** | **mass in components 960–7200 µm — the legibility score** |
| `mass_smear` | mass in components >7200 µm — the saturation failure mode |

---

## Reproducing

```bash
pip install -r requirements.txt

# 1. inventory the published predictions in the open bucket
python scripts/survey_predictions.py --out-csv data/predictions_index.csv

# 2. resolve each prediction's TRUE raster pitch from its zarr .zattrs
python scripts/fetch_pitch.py --index-csv data/predictions_index.csv \
    --cache-dir data/predictions --out-csv data/predictions_pitch.csv

# 3. score every window (downloads ~1.2 GB of ds8 JPEGs the first time)
python scripts/build_atlas.py --index-csv data/predictions_pitch.csv \
    --cache-dir data/predictions --out-csv data/atlas.csv

# 4. distribution, per-scroll table, contact sheets, decile ladder
python scripts/atlas_report.py --atlas-csv data/atlas.csv \
    --index-csv data/predictions_pitch.csv --cache-dir data/predictions \
    --out-dir figures

# 5. calibrate against the blind labels
python scripts/calibrate.py --key-csv data/labels/labeling_key.csv \
    --labels-csv data/labels/labels.csv --atlas-csv data/atlas.csv --at 0.900

# 6. per-segment maps, summary and hotspots
python scripts/segment_map.py --atlas-csv data/atlas.csv \
    --index-csv data/predictions_pitch.csv --cache-dir data/predictions \
    --out-dir figures --thresh 0.900
```

Steps 3–6 re-run from the local cache without re-downloading. Rebuilding the atlas from
cache is cheap; only step 3's first run costs bandwidth.

### Scripts

| script | what it does |
|---|---|
| `survey_predictions.py` | walks the open bucket, inventories published predictions |
| `check_pitch.py` | the metadata-free proof that `resolution_um` is not the pitch |
| `probe_meta.py` | small bucket/zarr metadata walker used to find `source_group` |
| `fetch_pitch.py` | resolves the true pitch per prediction, self-verifying against JPEG dims |
| `build_atlas.py` | scores every window — the core metric lives in `score_window()` |
| `spatial_structure.py` | the same metric applied to a single loose image |
| `atlas_report.py` | distribution, per-scroll table, contact sheets, decile ladder |
| `sample_for_labeling.py` | draws the blind stratified sample and renders label sheets |
| `rejoin_key.py` | re-attaches labels to a rebuilt atlas by pixel coordinates |
| `calibrate.py` | AUC, threshold sweep, corpus implication |
| `segment_map.py` | per-segment maps, `segment_summary.csv`, `hotspots.csv` |
| `scroll_diag.py` | per-scroll diagnostics — this is what caught the window-geometry bug |
| `recipe_ab.py` | paired recipe comparison with a sign test |

---

## Contributing / what would help most

1. **Re-label windows blind.** The protocol is in `sample_for_labeling.py`; the sheets carry
   only an index. Expert labels would move this from self-validated to externally validated,
   which is its weakest point by far.
2. **Attack a hotspot.** `data/hotspots.csv` and the zero-readable-window list in
   `data/segment_summary.csv` have full-resolution coordinates. If you improve a prediction
   there, this repo gives you a before/after number instead of an opinion.
3. **Break the metric.** If you can find a window that reads clearly and scores below 0.90,
   or a speckle field that scores above it, open an issue with the coordinates.

## License

Code: MIT (see [`LICENSE`](LICENSE)).
Derived data (`data/`, `figures/`, `results/`): CC BY 4.0.
The underlying scroll data and ink predictions belong to
[Vesuvius Challenge](https://scrollprize.org) and are used under their open-data terms.
