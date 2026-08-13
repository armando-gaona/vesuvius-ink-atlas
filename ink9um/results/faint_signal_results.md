# Does the weight soup buy clearness by losing faint ink?

Result of the pre-registered protocol in `../protocols/faint_signal_protocol.md`, written before any
of the numbers below existed. Run 2026-08-12. No GPU, no downloads: every prediction was already on
disk from the soup experiment.

**Headline, in the words the protocol fixed in advance: NOT RESOLVED.** The hardest-quartile recall
delta is positive on two segments and negative on one, which the reading rule declares a mixed
result, and a mixed result is not narrated into a direction.

What *is* resolved, and is worth more than the primary: **the soup is measurably smoother, and the
AUC gain is not a pure ranking artifact.** It survives at a matched operating point in 5 of the 6
runs.

## Where the question comes from

A Vesuvius team member, on the published before/after figure:

> one thing to be careful with ensembles is that sometimes you gain a lot of "clearness", like
> smoothing out the letters, but you can also lose fainter signals and generalization

The published AUC is computed over every pixel of the held-out region, so it is dominated by the
bulk of the ink. If averaging improves the bulk while degrading the few pixels carrying weak ink,
the headline rises for the wrong reason. This measures that directly.

## Gate

The published held-out AUCs were recomputed from the exact files used here, before any new quantity
was computed. All twelve reproduce.

| segment | base s42 | base s43 | soup s42 | soup s43 |
|---|---|---|---|---|
| pherc0814-46527 | 0.8683 | 0.8433 | 0.8692 | 0.8630 |
| pherc0139-w016 | 0.8014 | 0.9087 | 0.8525 | 0.8960 |
| pherc1667-w029 | 0.8717 | 0.9197 | 0.8966 | 0.9297 |

One cell differs in the fourth decimal: `pherc0139-w016` soup seed42 reads 0.8525 here against 0.8526
published. That is rounding of the same number, not a different file, and it is inside the declared
tolerance. Everything else is exact. This confirms the `_f64` soup builds were used, which is the
whole point of the gate: several soup builds of the same soup exist on disk.

Held-out region sizes, which set the ceiling on everything below: 161,051 px (0814), 175,222 px
(w016), 382,353 px (1667). Ink fraction 37.33% / 23.52% / 23.02%.

## Question 1: is the gain paid for on the hardest ink?

Recall inside each quartile of ink difficulty, at a threshold set so both models produce **the same
false positive rate on the non-ink pixels of the held-out region**. Difficulty comes from the
**other seed's** baseline, so the variable defining the strata is on neither side of the difference.
Stratum 0 is the hardest ink.

### The primary statistic

Hardest quartile, matched FPR 5%, mean of the two seeds, soup minus baseline:

| segment | seed42 | seed43 | **primary (mean)** |
|---|---|---|---|
| pherc0814-46527 | +0.0000 | +0.0098 | **+0.0049** (degenerate, see below) |
| pherc0139-w016 | +0.0384 | -0.1259 | **-0.0437** |
| pherc1667-w029 | +0.1006 | +0.0445 | **+0.0726** |

Two positive, one negative. Under the pre-registered reading rule that is neither "the objection
holds" nor "the objection does not hold". It is **not resolved**, and it is reported in those words.

### The degeneracy the protocol did not anticipate

On `pherc0814-46527` the primary is computed from a floor. At matched FPR 5%, recall in the hardest
quartile is **0.0000 for both the baseline and the soup on seed42**, and 0.0053 against 0.0151 on
seed43. At FPR 1% it is 0.0000 for all four runs. The +0.0049 is the difference between two numbers
that are almost zero, and it should not be read as a defence of the soup.

This is a finding in its own right and it is independent of the soup question: **on that segment,
at a usable operating point, neither model recovers any of the hardest quartile of ink.** It is not
universal (at FPR 5% the same cell reads 0.31/0.35 and 0.14/0.24 on the other two segments), so it
is a property of that segment, not of the models.

### The sign is not stable across the operating point

The primary was fixed at FPR 5% in advance. Here is what the other two declared levels give, so
that the choice cannot be mistaken for a robust one:

| segment | FPR 1% | FPR 5% (primary) | FPR 10% |
|---|---|---|---|
| pherc0814-46527 | +0.0000 | +0.0049 | **-0.0191** |
| pherc0139-w016 | -0.0047 | -0.0437 | -0.0793 |
| pherc1667-w029 | +0.0098 | +0.0726 | +0.0300 |

On `pherc0814-46527` the sign flips between 5% and 10%, driven by seed42 losing -0.0564 in the
hardest quartile at 10%. Only `pherc0139-w016` is consistent, and consistently negative.

### The geometric cross-check disagrees, and the disagreement is the result

The second stratifier never touches a prediction: distance from each ink pixel to the nearest
non-ink pixel, from the label alone. Stratum 0 is the thinnest and edge ink. At FPR 5%, soup minus
baseline:

| segment | seed | S0 (thinnest) | S1 | S2 | S3 |
|---|---|---|---|---|---|
| pherc0814-46527 | 42 | +0.0428 | +0.0673 | +0.0739 | +0.0611 |
| pherc0814-46527 | 43 | +0.0299 | +0.0927 | +0.0414 | +0.0271 |
| pherc0139-w016 | 42 | +0.0474 | +0.0485 | +0.0300 | +0.0257 |
| pherc0139-w016 | 43 | **-0.0519** | **-0.0429** | **-0.0449** | **-0.0337** |
| pherc1667-w029 | 42 | +0.0056 | +0.0374 | +0.1171 | +0.1786 |
| pherc1667-w029 | 43 | +0.0083 | +0.0168 | +0.0762 | +0.0426 |

The protocol says that if the two stratifiers disagree, the disagreement is reported rather than
resolved by preference. They do disagree, and specifically about `pherc0139-w016` seed43.

The cross-seed stratifier says that run loses -0.1259 in the hardest quartile, which reads as a
faint-ink-specific loss. The geometric stratifier says it loses roughly the same amount in **every**
stratum, from the thinnest ink to the thickest: -0.0519, -0.0429, -0.0449, -0.0337. A flat loss
across all strata is not selective damage to weak signal, it is a run that is simply worse.

That is consistent with what was already published: `pherc0139-w016` seed43 is the one of six runs
whose overall held-out AUC **drops** under the soup, 0.9087 to 0.8960. It is declared in three
places in `soup_results.md` and it is the same run again here.

This is stated as an observation, not as a rescue. The primary statistic is the primary statistic
and it is mixed. But anyone reading the negative cell should know that the exogenous stratifier does
not support the reading that the loss is concentrated on faint ink.

### Overall recall at a matched operating point, which the protocol asked for as context

The "all ink" column, FPR 5%, is the answer to the narrower question of whether the AUC gain is a
ranking artifact that evaporates at a threshold:

| segment | seed | base | soup | delta |
|---|---|---|---|---|
| pherc0814-46527 | 42 | 0.4604 | 0.5217 | +0.0613 |
| pherc0814-46527 | 43 | 0.4970 | 0.5448 | +0.0478 |
| pherc0139-w016 | 42 | 0.3592 | 0.3972 | +0.0380 |
| pherc0139-w016 | 43 | 0.6083 | 0.5650 | **-0.0433** |
| pherc1667-w029 | 42 | 0.5362 | 0.6209 | +0.0847 |
| pherc1667-w029 | 43 | 0.6568 | 0.6929 | +0.0361 |

Five of six gain, by +0.036 to +0.085, at an equal false positive rate. The one loss is the same
seed43 run on `pherc0139-w016`. The earlier note in `holdout9_results.md` that "averaging smooths,
and smoothing helps the ranking (AUC) more than the binary decision at a fixed threshold" is
therefore **not** the whole story: at a matched operating point the gain is still there, and it is
larger than the AUC gain it came from.

## Negative control

Everything above repeated on the **non-ink** pixels, stratified the same way. If the soup raised
scores on non-ink in the same pattern in which it raises them on faint ink, it would be firing more
everywhere and the matched FPR would be hiding it. Soup minus baseline, FPR 5%, stratum 3 is the
most confusable non-ink:

| segment | seed | S0 | S1 | S2 | S3 |
|---|---|---|---|---|---|
| pherc0814-46527 | 42 | +0.0000 | +0.0000 | +0.0000 | +0.0027 |
| pherc0814-46527 | 43 | +0.0000 | +0.0000 | -0.0013 | +0.0012 |
| pherc0139-w016 | 42 | -0.0012 | -0.0041 | +0.0002 | +0.0059 |
| pherc0139-w016 | 43 | +0.0089 | +0.0017 | -0.0078 | -0.0002 |
| pherc1667-w029 | 42 | +0.0066 | +0.0137 | +0.0051 | -0.0276 |
| pherc1667-w029 | 43 | +0.0058 | -0.0017 | -0.0057 | -0.0056 |

**The control passes.** Every cell is inside +-0.014 except one, `pherc1667-w029` seed42 stratum 3
at -0.0276, and that one goes in the direction of *fewer* false positives on the most confusable
non-ink.

Reported without softening: on `pherc1667-w029`, the segment where the soup gains most on faint ink,
seed42 also fires slightly more on the three easier non-ink strata (+0.0066, +0.0137, +0.0051). The
magnitudes are an order of magnitude below the +0.1006 gain in the hardest ink quartile of the same
run, so this does not invalidate the positive reading there, but it is in the table rather than
omitted from it.

## Question 2: is the soup output measurably smoother?

Each prediction is standardised to zero mean and unit variance over the held-out region first, so a
global change of scale or offset cannot register as a change of sharpness. Mean squared gradient
magnitude, over the held-out mask eroded by one pixel so the mask boundary does not enter:

| segment | seed | base | soup | ratio soup/base |
|---|---|---|---|---|
| pherc0814-46527 | 42 | 0.00345 | 0.00320 | 0.9256 |
| pherc0814-46527 | 43 | 0.00373 | 0.00373 | **1.0010** |
| pherc0139-w016 | 42 | 0.00633 | 0.00537 | 0.8496 |
| pherc0139-w016 | 43 | 0.00431 | 0.00404 | 0.9381 |
| pherc1667-w029 | 42 | 0.00509 | 0.00427 | 0.8397 |
| pherc1667-w029 | 43 | 0.00479 | 0.00416 | 0.8689 |

**The soup is smoother in five of the six runs**, by 6% to 16% of gradient energy. The sixth is
flat, not smoother.

The spectral secondary was **dropped on two of three segments** and its absence is stated here
rather than quietly omitted: the held-out region of `pherc0814-46527` and `pherc0139-w016` fills
only 70.1% and 60.8% of its own bounding box, below the 80% floor fixed in advance, and a spectrum
computed over a region that is 30% to 40% hole is a spectrum of the hole. Only `pherc1667-w029`
qualified, at 87.3% fill over 702x385 px. There, energy finer than 192 um:

| seed | base | soup | ratio |
|---|---|---|---|
| 42 | 0.0071 | 0.0036 | 0.5094 |
| 43 | 0.0034 | 0.0033 | 0.9807 |

The two seeds disagree by a factor of two. Seed42 loses half its fine-scale energy; seed43 loses
nothing measurable. With one segment and two seeds, no conclusion is drawn from this.

## Uncertainty

Block bootstrap of the primary statistic, resampling square tiles rather than pixels, both seeds
resampled on the same tiles, 1000 replicates. The threshold is recomputed inside each replicate,
because it is part of the statistic and freezing it would understate the spread.

| segment | block | n_blocks | CI95 | P(delta>0) |
|---|---|---|---|---|
| pherc0814-46527 | 128 px | 17 | [-0.0153, +0.0291] | 0.662 |
| pherc0814-46527 | 256 px | 6 | [-0.0443, +0.2492] | 0.667 |
| pherc0139-w016 | 128 px | 23 | [-0.1392, +0.0377] | 0.108 |
| pherc0139-w016 | 256 px | 10 | [-0.0960, +0.0653] | 0.164 |
| pherc1667-w029 | 128 px | 42 | [-0.0640, +0.1723] | 0.815 |
| pherc1667-w029 | 256 px | 14 | [-0.1062, +0.1719] | 0.717 |

**Four of the six lines have too few blocks to be an interval** (17, 6, 10 and 14 against the floor
of 20 declared in advance) and are printed as an order of magnitude, not a bound. Only
`pherc0139-w016` at 128 px and `pherc1667-w029` at 128 px clear it.

Every interval crosses zero. This was stated in advance and is repeated without retrofitting: no
single segment was expected to reach significance on its own. The published soup gain did not
either, and this is a smaller effect measured on a quarter of the ink pixels.

## The declared prediction, scored

Written before the run:

> The soup output will be measurably smoother, and it will not lose the hardest ink. Concretely:
> gradient energy lower for the soup in all six runs, and the hardest-quartile recall delta at or
> above zero in at least two of the three segments.

- First half: **failed.** Gradient energy is lower in five of six runs. `pherc0814-46527` seed43
  came out at 1.0010, flat. The prediction said all six and it was not all six.
- Second half: **held**, two of three segments at or above zero. But one of those two is the
  degenerate `+0.0049` on `pherc0814-46527`, so it held on a technicality and is reported as such.

Scoring it this way rather than rounding 1.0010 down to "essentially all six" is the point of
writing the prediction down.

## What this does and does not say

**Says.** On these three segments, the soup produces a smoother output, and the improvement it
buys is not confined to the ranking: it survives at a matched false positive rate in five of six
runs, with more overall recall for the same false positives. There is no measured evidence that it
fires more on non-ink. The one run where it loses does so uniformly across every stroke thickness,
which does not look like selective destruction of weak signal.

**Does not say.** It does not say the objection is wrong. The primary statistic is mixed and is
declared unresolved. It does not say anything about **cross-scroll generalisation**, which is the
other half of the objection: the `ink_9um` README confirms all 24 aligned segments went into
training, so no unseen scroll with annotation exists in public data, and every measurement here is
within-scroll, within-segment, held-out region. It does not say weight soups do or do not wash out
faint ink in general. It speaks about these published checkpoints on these three segments, at one
annotated plane.

**The binding limit is the data, not the design.** Three segments ship `_validation_mask.zarr` and
that is the entire public ceiling. Four of six bootstrap lines are below the block floor for the
same reason. Any of the ambiguities above would be settled by more held-out annotation, and by
nothing else.

## Reproducing

```
python ink9um/scripts/faint_signal.py <labels_dir> <predictions_dir> --resamples 1000
```

Refuses to run if the published AUCs do not reproduce. The full stdout of the run behind this file
is kept verbatim in `faint_signal_run.txt`, including the cells that are not quoted above.
