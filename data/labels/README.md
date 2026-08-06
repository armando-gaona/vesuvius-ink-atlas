# Blind labels — how to re-label, and why it matters

The weakest part of this project is that these labels were produced by an AI assistant under
human supervision, not by a papyrologist. Everything else is measurement; this is judgement.
**Independent re-labelling is the single highest-value contribution anyone can make here**,
and this directory is set up so it costs an hour and no downloads.

## The protocol

Labelling while looking at the score you intend to validate manufactures the answer. So:

- The sheets carry **an index and nothing else** — no score, no scroll name, no coordinates.
- Windows are **shuffled** before rendering, so sheet order carries no information.
- The index → window key (`labeling_key.csv`) was written **before any label existed** and is
  not needed until after the labels are in.
- The sample is **stratified** over 10 score bands, finer near the top where the decision
  boundary lives, and **round-robin across scrolls within each band** — PHercParis4 is 68% of
  the corpus, so a plain draw inside a band is a PHercParis4 draw.

## To re-label

1. Open `sheets_round1/` and `sheets_round2/`. Do **not** open `labeling_key.csv` first.
2. For each numbered tile, write one line in your own CSV: `i,label` with label being
   **`text`** (you can make out letterforms), **`partial`** (letterforms present but broken
   or drowned) or **`speckle`** (no letterforms).
3. Then run:

   ```bash
   python ../../scripts/calibrate.py --key-csv labeling_key.csv \
       --labels-csv your_labels.csv --atlas-csv ../atlas.csv
   ```

   It reports AUC, a full precision/recall sweep, and what your threshold implies for the
   whole corpus.
4. Open an issue or PR with your labels. Disagreement is a result, not a problem — if the AUC
   drops under expert labels, that is the most useful thing this repo could learn.

## Files

| file | what it is |
|---|---|
| `labeling_key.csv` | index → (scroll, segment, recipe, y0, x0, band, score), 125 rows |
| `labels.csv` | the labels, 160 rows |
| `sheets_round1/` | 30 sheets, first draw (120 windows) |
| `sheets_round2/` | 20 sheets, supplement (40 windows) |

**160 label rows but 125 join to the key.** A geometry bug fix — the analysis window had to
be defined in microns rather than pixels, see the repository README — moved the window grid
and orphaned 35 of the original labels. The orphans were, unavoidably, concentrated in
exactly the two arms the fix touched (PHerc0172 and the `1um_s1z2` recipe), so calibrating
without them would have verified everything except the correction. Round 2 re-drew 40
windows blind restricted to those arms (`--pitch-in 2.258,7.91,2.215`, indices starting at
201 so they could not collide). The orphaned rows are kept rather than deleted, because a
label set that quietly loses its inconvenient members is not auditable.
