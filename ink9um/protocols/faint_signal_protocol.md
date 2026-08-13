# Does the weight soup buy clearness by losing faint ink? (pre-registered 2026-08-12)

Written BEFORE computing anything. The only numbers that exist at the time of writing are the ones
already published in `../results/`. No new quantity has been looked at.

## Where the question comes from

A Vesuvius team member, reacting to the published before/after figure, raised a specific objection:

> one thing to be careful with ensembles is that sometimes you gain a lot of "clearness", like
> smoothing out the letters, but you can also lose fainter signals and generalization

That is a serious criticism of the result in `soup_results.md`, and it is not answered by anything
measured so far. The reason it bites is structural: **the reported AUC is computed over every pixel
of the held-out region, so it is dominated by the bulk of the ink.** Strong, thick, obvious ink
contributes most of the pixels. If averaging weights improves that bulk while degrading the small
number of pixels carrying weak ink, the headline AUC rises while the thing that actually matters for
discovery gets worse. A metric can move in the right direction for the wrong reason.

There is also a warning already sitting in our own results that was noted and not chased:
`holdout9_results.md` records that on F1 the averaged prediction does not always win, with the
observation that "averaging smooths, and smoothing helps the ranking (AUC) more than the binary
decision at a fixed threshold". That is a shadow of the same effect. This protocol chases it.

## What is being tested, in one sentence

Whether the soup's published gain of +0.010 to +0.019 AUC is paid for with recall on the hardest
ink, and whether the prediction it produces is measurably smoother than the baseline's.

Those are two separate questions and they are kept separate on purpose. A prediction can be smoother
without being blinder, and the interesting answer is the combination of the two.

## Data, fixed now

No GPU and no downloads. Everything is already on disk.

- Segments: `pherc0814-46527`, `pherc0139-w016`, `pherc1667-w029`. The three that ship
  `_validation_mask.zarr`. This is the ceiling of public data, not a choice.
- Plane Z=10, the only annotated plane of the aligned set.
- Region: held out only, defined exactly as in the existing scripts, `validation_mask & ~supervision_mask`.
- Pitch: 9.596 um/px. Every spatial quantity below is declared in microns and converted with this
  number, never assumed in pixels.
- Configurations compared: the **A4 soup** (average of the last four published steps of one seed)
  against the **step-075000 baseline**, per seed, exactly the pair that produced the published table.
- Both seeds, 42 and 43. Per-seed results are reported and the primary statistic is their mean.

**Nothing else is added.** No TTA, no other soup, no other step. Mixing them in is how a comparison
turns into a search.

## Gate before any new analysis

The published held-out AUCs are recomputed from the exact files that will be used, and compared
against `soup_results.md`: baseline means 0.8558 / 0.8550 / 0.8957 and A4 means 0.8661 / 0.8743 /
0.9131.

If they do not reproduce to the fourth decimal, **the run stops** and the file selection is wrong.
Several soup builds exist on disk, including float32 and float64 accumulations of the same soup, and
picking the wrong one would silently answer a different question. This gate is cheap and it removes
the possibility.

## Question 1: is the gain paid for on the hardest ink?

### Defining "hard ink" without letting the model define it

Ranking ink pixels by the confidence of the very model being judged would guarantee the answer. Two
stratifiers are declared, one primary and one as an exogenous cross-check.

**Primary, cross-seed.** Ink pixels are ranked by the **baseline prediction of the other seed** and
split into quartiles. The quantity then measured on this seed is `soup minus baseline`, so the
variable defining the strata is produced by a model that appears on neither side of that difference.

Stated limitation, declared now rather than discovered later: the two seeds share architecture and
training data, so their errors are correlated and this stratifier is not fully exogenous. It is
targeted at exactly the right notion of difficulty, which is why it is primary, and the second
stratifier exists to catch the case where that correlation is driving the result.

**Secondary, purely geometric.** Distance from each ink pixel to the nearest non-ink pixel, from the
label alone, never touching any prediction. This separates stroke interiors from stroke boundaries.
It is fully exogenous and it is a blunter proxy: it measures thinness, not faintness. Reported
alongside, and if the two stratifiers disagree, that disagreement is the result and is reported as
such rather than resolved by preference.

### The operating point is matched, and this is the load-bearing detail

Recall cannot be compared between two models at the same numeric threshold. A model whose outputs
are simply shifted upward would "recover more faint ink" for free, which is not a finding, it is an
artifact of scale.

So the threshold for each prediction is chosen to produce **the same false positive rate on the
non-ink pixels of the held-out region**, and recall is compared at that matched rate. Levels fixed
in advance: **FPR 1%, 5% and 10%**, with **5% as the primary**.

### Primary statistic

Recall inside the **hardest quartile of ink**, at **matched FPR 5%**, with difficulty from the
**cross-seed** stratifier, averaged over the two seeds, reported per segment. The comparison is soup
minus baseline.

Recall in the other three quartiles is reported in the same table, because the shape across
quartiles is more informative than any single cell, but it is not the primary.

### Reading rule, fixed now

Margin: **0.01 of absolute recall**, the same margin used by the two previous sweeps and the soup
protocol.

- **The objection holds**: the hardest-quartile delta is negative in all three segments, or negative
  beyond the margin in two and not positive beyond it in the third. This is published as such, and it
  qualifies the recommendation in `soup_results.md`.
- **The objection does not hold**: the hardest-quartile delta is zero or positive in all three
  segments. Published as a measured defence, with the interval, not as a clean bill of health.
- **Anything else is "not resolved"**, and is reported in those words. With three segments a mixed
  result is a mixed result and will not be narrated into a direction.

### Uncertainty

The same block bootstrap already used in `bootstrap_auc.py`, resampling square tiles rather than
pixels, both seeds resampled on the same tiles, 1000 replicates, block sizes 128 and 256 px, and the
number of available blocks printed. Fewer than 20 blocks is reported as a warning and not as an
interval.

It is stated in advance and without waiting for the numbers: **no single segment is expected to
reach significance on its own.** The published soup gain did not, and this is a smaller effect
measured on a subset of the pixels. The evidence here is direction and replication across three
segments, and the write-up will say that in those words rather than leaning on an interval that
cannot carry it.

## Question 2: is the soup output measurably smoother?

"Smoothing out the letters" is not an impression, it is a physical operation that removes fine
detail. It is measured directly.

Each prediction is standardised to zero mean and unit variance **over the held-out region** before
anything is measured, so that a global change of scale or offset cannot register as a change of
sharpness. Then:

- **Primary**: mean squared gradient magnitude over the held-out region, soup against baseline.
  Smoothing lowers it.
- **Secondary**: radially averaged power spectrum, reported as the fraction of energy above
  **192 um** (20 px at this pitch), a scale finer than a stroke and therefore where blurring shows
  first. If the held-out region is too small to support a stable spectrum, the secondary is dropped
  and its absence is stated, not quietly omitted.

There is no reading rule attached to this one because there is no threshold at which "smoother" is
good or bad. It is a descriptive measurement whose whole purpose is to be crossed with question 1.

## Mandatory negative control

Everything in question 1 is repeated on the **non-ink** pixels of the held-out region, stratified the
same way.

If the soup raises scores on non-ink pixels in the same pattern in which it raises them on faint ink,
then it is not recovering weak signal, it is firing more everywhere, and the matched FPR is hiding it
somewhere else. That outcome invalidates the positive reading of question 1 and is reported as such.

This is the same control that governs the pair normalisation experiment, for the same reason: a
manipulation that manufactures signal must not be allowed to look like a manipulation that recovers it.

## Declared prediction

Written now so that it counts as a prediction rather than an explanation produced afterwards.

**The soup output will be measurably smoother, and it will not lose the hardest ink.** Concretely:
gradient energy lower for the soup in all six runs, and the hardest-quartile recall delta at or above
zero in at least two of the three segments.

The reasoning, so that the prediction can be judged and not just scored: the mechanism measured in
`soup_results.md` is variance reduction across the seed lottery, seed separation falling from 0.0601
to 0.0276, and averaging two disagreeing models produces a flatter output almost by construction. But
variance reduction should help most where the seeds disagree, and there is no measured reason for it
to specifically destroy weak signal.

**Where this prediction can fail, and what that would mean.** If gradient energy drops *and* the
hardest quartile loses recall, the objection is correct and the honest statement becomes: the soup
improves the ranking metric by sharpening what is already visible, and should not be used for
discovery. That result would be published with the same weight as a favourable one, and it would be
more useful to the project than a defence.

## What this experiment cannot say

It cannot say anything about **cross-scroll** generalisation, which is the other half of the team
member's comment. The `ink_9um` README confirms that all 24 aligned segments went into training, so
no unseen scroll with annotation exists in public data. Every measurement here is within-scroll,
within-segment, held-out region. That limit is a property of the available data and it is stated in
the write-up rather than glossed.

It also cannot say that weight soups do or do not wash out faint ink in general. It speaks only about
these published checkpoints on these three segments.

## Multiplicity, stated plainly

This design produces 3 segments times 2 seeds times 4 quartiles times 3 FPR levels times 2
stratifiers, which is far more cells than there is evidence to support. That is why the primary
statistic is named above and fixed to a single cell definition. Everything else is descriptive
context. No cell other than the declared primary will be promoted into a headline afterwards, and the
full table is published so that anyone can see what was not promoted.
