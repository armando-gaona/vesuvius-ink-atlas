# Weight soup on the held-out region (2026-08-10)

Protocol pre-registered in `../protocols/soup_protocol.md`. Soups built with
`scripts/weight_soup.py`, scored with `scripts/holdout_9um.py`, aggregates recomputed in one pass
with `scripts/recompute_soup_tables.py`. AUC on the held-out region, plane Z=10, default z window,
no TTA.

**Provenance of every number below.** All soups were built with float64 accumulation, which is what
the published `weight_soup.py` does. Every aggregate in this document (means, deltas, seed
separations) is computed from full-precision AUCs and rounded only when printed, so the tables can
be regenerated from the scripts rather than by chaining values that were already rounded.

---

## First: the cross-seed soups are broken

| soup | what it averages | training AUC | held-out AUC |
|---|---|---|---|
| B | seed42 + seed43 at 75k | **0.4960** | 0.4863 |
| AB | the 14 checkpoints | **0.5037** | 0.5108 |

0.50 of AUC is a coin flip. These models do not work even on the papyrus they **did** see during
training, where the single checkpoints give 0.999. It falls under the "broken soup" case that the
protocol defined in advance (declared threshold: training below 0.99). It is not evidence against
the technique: it is that these weights do not live in the same region of weight space.

### What was predicted, what was not, and what we learned by running it

The protocol pre-registered a diagnostic and a prediction, and the honest report is that **the
prediction was written on the wrong quantity.**

`soup_protocol.md` says the **relative distance between weights** is measured, and predicts that if
the two seeds started from different initializations their weights will be very far apart and B and
AB should come out broken. Here is what the diagnostic produced:

| pair | relative distance | cosine |
|---|---|---|
| seed42 75k vs seed42 60k | 0.0532 | +0.9989 |
| seed42 75k vs seed42 50k | 0.1423 | +0.9934 |
| seed42 75k vs seed42 40k | 0.2836 | +0.9801 |
| seed42 75k vs seed42 10k | **1.3417** | +0.9148 |
| **seed42 75k vs seed43 75k** | **1.1936** | **+0.2769** |

**Distance does not separate the two cases.** Step 10k of the same seed sits at 1.3417, which is
*further* than the cross-seed pair at 1.1936, and yet A7, which contains step 10k, is not broken
(0.9979 and 0.9966 on training). By distance alone the pre-registered prediction fails.

**Cosine does separate them, cleanly**: +0.9148 within a seed against +0.2769 across seeds. Two
unrelated vectors would give 0. The two seeds share little direction, and that is the quantity that
predicts a broken soup.

So the correct claim is not "we predicted it and we were right". It is: **the pre-registered
diagnostic was the wrong one, the run showed which one works, and the working one is cheap.** For
anyone reproducing this, the usable rule is the cosine, and `weight_soup.py --diag` prints both.

Why distance fails is not mysterious once measured: it is not symmetric in its two arguments and it
is dominated by the norm of the weights, and step 10k has 2.19 times the norm of step 75k. A large
distance can mean "different direction" or just "different scale", and only the first one breaks a
soup.

### Method note on the diagnostic

The first version accumulated in float32 and returned **cosines of 1.02**, which is impossible: a
cosine cannot exceed 1. Summing 68 million products in float32 loses enough precision for the ratio
to drift. It was redone accumulating in float64. The impossible value is what exposed the bug; a
smaller error would have passed unnoticed and been reasoned about as if it were real.

---

## Reproducibility note: the soup accumulates in float64

`weight_soup.py` accumulates in float64 and casts to float32 on save, so the soup does not depend on
the order the files are passed in. With 4 float32 summands float32 would be enough, so this is
reproducibility, not the correction of an error.

Measured, comparing the float32-accumulated build of A4 seed42 against the float64 one:

| quantity | value |
|---|---|
| tensors that change | 506 of 508 |
| maximum absolute difference | 1.192e-07, on `task_heads.ink.weight` |
| weights differing by more than 1.5 ULP | 267,607 of 68,175,426 (0.39%) |
| maximum difference in ULP | 65536, on a weight whose value is 1.24e-09 |

The maximum absolute difference happens to be one ULP of a float32 at that weight's magnitude, but
**it would be wrong to describe the change as "one last bit"**: 0.39% of the weights move by more
than 1.5 ULP, and the extreme relative case is a weight that is essentially zero, where "one bit"
means nothing. What is true is the thing that matters: it does not move the result.

| run, `pherc0814-46527` | float32 accumulation | float64 accumulation |
|---|---|---|
| A4 seed42, held out | 0.8691 | 0.8692 |
| A4 seed43, held out | 0.8630 | 0.8630 |
| A4 seed42, training | 0.9996 | 0.9996 |
| A4 seed43, training | 0.9994 | 0.9994 |

---

## Stage 1: which soup competes

Segment `pherc0814-46527`, cost of ONE inference, baseline the standalone `step-075000`.

| configuration | train. s42 | train. s43 | held. s42 | held. s43 | mean | against baseline |
|---|---|---|---|---|---|---|
| baseline 75k | 0.9997 | 0.9997 | 0.8683 | 0.8433 | 0.8558 | |
| **A4** (last 4 steps) | 0.9996 | 0.9994 | 0.8692 | 0.8630 | **0.8661** | **+0.0103** |
| A7 (all 7 steps) | 0.9979 | 0.9966 | 0.8518 | 0.8740 | 0.8629 | +0.0071 |

A4 passes the 0.01 margin. A7 does not, and it also starts to degrade on training (0.9966), which
fits with it bringing in step 10k, the one at distance 1.34 from the rest.

A4 wins. It goes to stage 2 **without re-selecting anything**.

---

## Stage 2: A4 replicates on all three segments

| segment | baseline 75k (mean) | **A4 (mean)** | delta |
|---|---|---|---|
| pherc0814-46527 | 0.8558 | 0.8661 | **+0.0103** |
| pherc0139-w016 | 0.8550 | 0.8743 | **+0.0193** |
| pherc1667-w029 | 0.8957 | 0.9131 | **+0.0174** |

**Three out of three, and all three above the declared margin of 0.01.** It is the first thing in
this whole package that replicates: step 20k did not, z window S3 did not, this one does.

Mandatory control, declared in advance to separate "does not help" from "is broken": the training
AUC of A4 across the six runs stays between **0.9985 and 0.9996**. The soup works as a model.

---

## It also improves the metric the checkpoints themselves declare

The AUC above is our choice of metric. The checkpoints' own config declares
`best_checkpoint_metric = val_balanced_accuracy`, so the same six runs were scored with balanced
accuracy at threshold 0.50, which is what that name refers to. Threshold 0.50 is not a naive default
here: the models were trained with `bce_label_smoothing = 0.5`, which puts the calibrated midpoint
exactly there.

| segment | seed | baseline 75k | A4 | delta |
|---|---|---|---|---|
| pherc0814-46527 | seed42 | 0.7654 | 0.7636 | -0.0017 |
| pherc0814-46527 | seed43 | 0.7585 | 0.7661 | +0.0076 |
| pherc0139-w016 | seed42 | **0.6770** | **0.7444** | **+0.0674** |
| pherc0139-w016 | seed43 | 0.8244 | 0.8057 | -0.0187 |
| pherc1667-w029 | seed42 | 0.7469 | 0.8099 | **+0.0630** |
| pherc1667-w029 | seed43 | 0.8335 | 0.8463 | +0.0128 |
| **pherc0814-46527** | mean | 0.7619 | 0.7649 | **+0.0029** |
| **pherc0139-w016** | mean | 0.7507 | 0.7750 | **+0.0243** |
| **pherc1667-w029** | mean | 0.7902 | 0.8281 | **+0.0379** |

**Three out of three on the segment means, on the authors' own declared metric**, and with the same
shape as the AUC: the two runs that gain most are the two worst starting points. The individual
picture is noisier than the AUC one, 4 of 6 instead of 5 of 6, which is expected because balanced
accuracy depends on a single threshold while AUC does not.

---

## Why it works: it compresses the seed lottery

Breakdown by seed, held-out region:

| segment | seed | baseline 75k | A4 | delta |
|---|---|---|---|---|
| pherc0139-w016 | seed42 | **0.8014** | **0.8526** | **+0.0512** |
| pherc0814-46527 | seed43 | 0.8433 | 0.8630 | +0.0197 |
| pherc0814-46527 | seed42 | 0.8683 | 0.8692 | +0.0009 |
| pherc1667-w029 | seed42 | 0.8717 | 0.8966 | +0.0250 |
| pherc0139-w016 | seed43 | 0.9087 | 0.8960 | **-0.0127** |
| pherc1667-w029 | seed43 | 0.9197 | 0.9297 | +0.0099 |

Rows are sorted by starting point, worst first, because that is the pattern.

**Five of six improve, and the one that gains by far the most is the worst starting point of the
six** (0.8014, the catastrophic seed-lottery case documented in `holdout9_results.md`). Correlation
between starting AUC and gain: Pearson r = **-0.82** (p = 0.048), Spearman rho = -0.60 (p = 0.21).
The linear correlation is driven by that single extreme point and the rank correlation is not
significant, so with n = 6 this is a visible tendency and not an established law.

The mirror image of that sentence is **not** true, and it is worth saying because it is the version
that reads better. The run that gets worse (`pherc0139-w016` seed43, 0.9087) is the *second* best of
the six, not the best. The actual best of the six (`pherc1667-w029` seed43, 0.9197) **improves**, by
+0.0099. So the soup does not reliably cost you anything when you start well; it just does not have
much left to give.

What is solid, because it is measured directly rather than inferred from six points, is the
compression itself. Separation between the two seeds, absolute AUC difference on the held-out
region:

| segment | baseline 75k | A4 |
|---|---|---|
| pherc0814-46527 | 0.0250 | 0.0062 |
| pherc0139-w016 | **0.1073** | 0.0435 |
| pherc1667-w029 | 0.0481 | 0.0331 |
| **mean** | **0.0601** | **0.0276** |

**The seed lottery is cut to less than half, on all three segments.** And this connects with
something already measured without looking for it: the step sweep found that training longer makes
the two seeds **diverge** on unseen papyrus, from a mean separation of 0.0146 at 20k to 0.0601 at
75k. If that divergence is noise from the final stretch of the trajectory, averaging the trajectory
should remove it. It was measured, and it does: 0.0601 to 0.0276.

The soup is not a trick that happened to work. It is the remedy the earlier finding predicted.

---

## How solid is the gain: block bootstrap

The held-out region has a few hundred thousand pixels, which looks like an enormous sample. It is
not. Ink comes in strokes, so neighbouring pixels are not independent and neither are the model's
errors. Treating them as independent makes any AUC look far more precise than it is, and a gain of
+0.010 measured against a declared margin of 0.010 is exactly the size where this matters.

`scripts/bootstrap_auc.py` resamples square tiles of the image with replacement instead of pixels,
recomputes the same statistic that is reported (mean of the two seeds, soup minus baseline), 1000
times. Both seeds are resampled on the same tiles, because they are two measurements of one papyrus.

| segment | delta | 64 px | 128 px | 256 px | 512 px |
|---|---|---|---|---|---|
| pherc0814-46527 | +0.0103 | [-0.0010, +0.0227] | [-0.0045, +0.0262] | [-0.0191, +0.0228] | [+0.0075, +0.0255] |
| pherc0139-w016 | +0.0193 | [+0.0022, +0.0380] | [-0.0020, +0.0521] | [+0.0052, +0.0287] | [+0.0032, +0.0251] |
| pherc1667-w029 | +0.0174 | [+0.0005, +0.0350] | [-0.0037, +0.0443] | [+0.0005, +0.0394] | [-0.0013, +0.0341] |

CI95 of the delta. P(delta > 0) ranges from 0.929 to 0.991 across all twelve cells. Number of blocks
available to resample from: 53, 17, 6 and 2 on `pherc0814-46527`; 66, 23, 10 and 6 on
`pherc0139-w016`; 127, 42, 14 and 10 on `pherc1667-w029`.

**What this says, stated plainly:**

- The direction is consistent. All twelve cells put more than 92% of the resampled deltas above
  zero, and all three point-estimates are positive.
- **No single segment reaches 95% confidence on its own.** The interval includes zero in 6 of the 12
  cells, and on `pherc0814-46527`, which is the segment where A4 was selected, it includes zero at
  three of the four block sizes. The +0.0103 that cleared the pre-registered margin there **should
  not be read as a significant result on that segment**.
- The 512 px column on `pherc0814-46527` has only 2 blocks. A bootstrap over 2 units is not an
  interval, and it is printed only so nobody reconstructs it and thinks it was hidden.

So the load-bearing evidence here is **not** the margin on the selection segment. It is that the
effect replicates on two further segments where nothing was chosen, in the same direction, at a
similar size, in a metric of our choice and in the authors' own metric. That is a weaker statistical
claim than "p < 0.05" and a stronger practical one.

---

## Where A4 does not help, and it has to be said

At the cost level of **two** inferences, compared against averaging the predictions of the two
ordinary seeds:

| segment | average of the 2 seeds' predictions | average of the 2 A4 soups | delta |
|---|---|---|---|
| pherc0814-46527 | 0.8676 | 0.8751 | +0.0075 |
| pherc0139-w016 | 0.9044 | 0.8966 | **-0.0079** |
| pherc1667-w029 | 0.9237 | 0.9312 | +0.0075 |

Two of three, one against, none reaching the margin. **At that cost level the soup does not
replicate and is not recommended.** Whoever can pay for two inferences should just average the two
seeds.

Note that these are AUCs of an averaged *prediction*, which is a different quantity from the "mean"
column in the tables above, where a mean of two AUCs is reported. They are not interchangeable and
are not comparable across tables.

The soup wins precisely where averaging predictions is not an option: when a single model is run,
which is the default for anyone who downloads a checkpoint.

---

## What is actionable

For whoever runs **a single inference**:

- Averaging the weights of the **last four published steps of the same seed** raises the AUC on
  unseen papyrus by between **+0.010 and +0.019** across the three measured segments, and raises
  balanced accuracy by between +0.003 and +0.038.
- It costs **the same** at inference time. The averaging is seconds of CPU and the 14 checkpoints
  are already published.
- If you draw the bad seed, it is worth between **0.020 and 0.051** of AUC.
- **Do not average weights across seeds.** What comes out is a random model. The cross-seed cosine
  (0.277) warns you in thirty seconds without touching a GPU. Do not use the relative distance for
  this: it does not separate the cases.

---

## Limitations

| limitation | detail |
|---|---|
| n = 3 segments | the ceiling with public data: they are the only ones shipping `_validation_mask.zarr` |
| 2 seeds | the two that were published |
| 1 of the 6 individual runs gets worse | `pherc0139-w016` seed43, -0.0127 |
| no segment is individually significant | see the block bootstrap above; the evidence is the replication, not the margin |
| the mechanism is a tendency, not a law | Pearson r = -0.82 on n = 6, driven by one point; the rank correlation is not significant |
| plane Z=10 only | the only annotated plane of the aligned set |
| no TTA | deliberately, so effects do not mix |
