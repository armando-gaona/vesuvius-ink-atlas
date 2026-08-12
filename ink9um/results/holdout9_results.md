# Seen region against held-out region, within the same segment (2026-08-10)

9 um models published on 2026-08-09 (`seed42_step-075000.pth`, `seed43_step-075000.pth`).
Label set `ink_9um/labels/aligned-scrollprizeorg-21slices`, plane Z=10 (the only annotated one of the
21). Script: `scripts/holdout_9um.py`.

## Why this comparison and not another

On the five segments with native annotation the AUC comes out above 0.995. That number says nothing:
those segments are the very annotated set that gets published, and the `ink_9um` README confirms all
24 aligned segments went into training, so the model saw them.

Three segments also come with `_validation_mask.zarr`, which marks the held-out region on which the
published checkpoint reports its metrics. The two regions are on the SAME papyrus, the SAME scan, the
SAME window in z and the SAME annotation. The only thing that changes is whether the model saw them
during training. The AUC drop between the two is the honest measure of how much of that 0.99 was
memory.

The two masks are **disjoint**, not nested: the measured overlap is 0.000%.

## Results

| segment | region | seed42 AUC | seed43 AUC | mean of the two AUCs |
|---|---|---|---|---|
| pherc0139-w016 | supervised | 0.9994 | 0.9996 | 0.9995 |
| pherc0139-w016 | **HELD-OUT** | **0.8014** | **0.9087** | **0.8550** |
| pherc0814-46527 | supervised | 0.9997 | 0.9997 | 0.9997 |
| pherc0814-46527 | **HELD-OUT** | **0.8683** | **0.8433** | **0.8558** |
| pherc1667-w029 | supervised | 0.9996 | 0.9992 | 0.9994 |
| pherc1667-w029 | **HELD-OUT** | **0.8717** | **0.9197** | **0.8957** |

The last column is the **mean of the two AUCs**, which is the expected AUC of picking one of the two
published seeds at random. It is not the same quantity as the AUC of the averaged prediction, which
is what the recommendation section below reports and which is higher. Everywhere in this package,
"mean" in a table of this shape means the mean of two AUCs, and the averaged prediction is always
named explicitly.

Best F1 on the held-out region: w016 0.558 / 0.717 / 0.716; 0814 0.733 / 0.695 / 0.737;
1667 0.655 / 0.762 / 0.738 (seed42 / seed43 / averaged prediction).

## Mandatory control: the drop is not explained by the amount of ink

The obvious objection is that the held-out region is simply worse papyrus, with less ink, and that
the drop has nothing to do with having seen it or not. Measured (seed42, plane Z=10):

| segment | ink in training | ink in held-out | held-out AUC |
|---|---|---|---|
| pherc0139-w016 | 21.40% | 23.52% | 0.8014 |
| pherc0814-46527 | 29.61% | 37.33% | 0.8683 |
| pherc1667-w029 | 29.37% | 23.02% | 0.8717 |

In two of the three, the held-out region has **more** ink than the training one, not less. And the
third one closes the argument: `pherc0139-w016` and `pherc1667-w029` have practically the same
proportion of held-out ink (23.5% and 23.0%) and yet give very different AUCs (0.80 and 0.87).
The amount of ink does not order the results, so it is not what explains them.

## How it degrades: the model does not get it wrong, it becomes undecided

| segment | region | mean where there IS ink | mean where there is NOT |
|---|---|---|---|
| pherc0139-w016 | supervised | 0.7795 | 0.2316 |
| pherc0139-w016 | HELD-OUT | 0.5010 | 0.3063 |
| pherc0814-46527 | supervised | 0.7757 | 0.2314 |
| pherc0814-46527 | HELD-OUT | 0.5719 | 0.3047 |

The two populations do not cross, they **move closer**: the ink one drops from 0.78 to 0.50-0.57 and
the non-ink one rises from 0.23 to 0.31. The model still points in the right direction but with far
less confidence. That is exactly the signature of memorizing instead of generalizing: on what it saw
it separates with confidence, on what it did not see it hesitates.

## The threshold moves, but gains little (measured, not assumed)

I first wrote that the default threshold of 0.50 was badly calibrated and that lowering it to 0.35
was a free improvement. I measured it and **it does not hold up as a general improvement**:

| segment | seed | best F1 (threshold) | F1 at 0.50 | gain |
|---|---|---|---|---|
| pherc0139-w016 | seed42 | 0.558 (0.35) | 0.509 | +0.049 |
| pherc0139-w016 | seed43 | 0.717 (0.50) | 0.717 | 0.000 |
| pherc0814-46527 | seed42 | 0.733 (0.35) | 0.702 | +0.031 |
| pherc0814-46527 | seed43 | 0.695 (0.45) | 0.692 | +0.003 |

Only two out of four cases gain anything, and they are precisely the two where the model degrades the
most. So the threshold shift is not an adjustment that is worth anything in itself: it is a
**symptom** of how far confidence has collapsed in that segment. Where the model holds up (seed43 on
w016), 0.50 is already optimal.

On the training region the threshold is irrelevant in all four cases (F1 at 0.50 within 0.001 of the
best). Recalibrating by looking at the supervised region would not have detected anything.

## The two findings

**1. The 0.999 is memory.** The real AUC outside what the model saw is between 0.80 and 0.91.
It is still a useful model, but the difference between 0.999 and 0.85 is the difference between "this
is solved" and "this half works".

**2. Which of the two seeds is better changes depending on the segment.** seed43 wins on
`pherc0139-w016` (0.9087 against 0.8014, eleven points) and on `pherc1667-w029` (0.9197 against
0.8717); seed42 wins on `pherc0814-46527` (0.8683 against 0.8433). Meanwhile the AUC on the
supervised region is identical to the third decimal place in all three cases (0.9992 to 0.9997).

This is what is actionable: **the AUC on the supervised region cannot be used to choose between two
published checkpoints, because a difference of eleven points on new papyrus leaves no trace in it.**
Any comparison between these checkpoints has to be made on a held-out region, which on public data
means one of these three segments. That is also where the authors say they report their own metrics,
so this is a statement about which measurement is informative, not a claim that anyone measured the
wrong thing.

And it is not enough to always keep seed43 just because it wins two out of three: on the third one
it loses, and there is no way of knowing in advance which case you are in.

## Measured recommendation: average the two predictions

Averaging the two outputs (no retraining, two inferences and a mean) costs twice the GPU and removes
the gamble:

| segment | worse seed | better seed | averaged prediction | against the better | against the worse |
|---|---|---|---|---|---|
| pherc0139-w016 | 0.8014 | 0.9087 | 0.9044 | -0.0043 | +0.1030 |
| pherc0814-46527 | 0.8433 | 0.8683 | 0.8676 | -0.0007 | +0.0243 |
| pherc1667-w029 | 0.8717 | 0.9197 | 0.9237 | **+0.0040** | +0.0520 |

The averaged prediction stays within less than 0.005 of the better seed in all three, and on one it
beats it.
Picking a seed blindly costs you between 0.024 and 0.103 of AUC half of the time. **Averaging costs
0.004 in the worst case and never leaves you on the bad side.**

Honest caveat: on F1 the averaged prediction does not always win. On `pherc1667-w029` it gives 0.738
against the 0.762 of seed43. Averaging smooths, and smoothing helps the ranking (AUC) more than the
binary decision at a fixed threshold.

## Mirror TTA: it helps a single seed, it does not add on top of the mean

`koine_machines.inference.infer` ships `--tta-mirror` off by default: it averages the prediction over
the flips of the spatial axes. It costs about 4 passes instead of 1.

| segment | seed | baseline | with TTA | delta |
|---|---|---|---|---|
| pherc0139-w016 | seed42 | 0.8014 | 0.8874 | **+0.0860** |
| pherc0139-w016 | seed43 | 0.9087 | 0.8911 | -0.0176 |
| pherc0814-46527 | seed42 | 0.8683 | 0.8510 | -0.0173 |
| pherc0814-46527 | seed43 | 0.8433 | 0.8624 | +0.0191 |
| pherc1667-w029 | seed42 | 0.8717 | 0.9244 | **+0.0527** |
| pherc1667-w029 | seed43 | 0.9197 | 0.9396 | +0.0199 |

Positive in 4 out of 6, mean **+0.024**. As an improvement to a single checkpoint, it works.

But **it does not add on top of the seed average**. Both columns are AUCs of an averaged prediction
of the two seeds, without and with TTA:

| segment | averaged prediction | averaged prediction with TTA | delta |
|---|---|---|---|
| pherc0139-w016 | 0.9044 | 0.8974 | -0.0070 |
| pherc0814-46527 | 0.8676 | 0.8646 | -0.0030 |
| pherc1667-w029 | 0.9237 | 0.9432 | +0.0195 |

Mean +0.003, that is to say nothing. The reading is that **the two things do the same job**: reducing
the variance of an unstable prediction. Once the two seeds have been averaged, TTA no longer has any
variance left to remove. Whoever can only afford one inference should use TTA; whoever can afford
two is better off running the two seeds than one seed with TTA.

**A hypothesis of mine that the measurement knocked down.** With the first four points the effect
looked perfectly monotonic: the worse the baseline, the more TTA helped. With all six it breaks
(`pherc1667-w029` seed43 starts from 0.9197, already high, and rises just the same). The sign of the
effect **cannot be predicted** from the baseline. It is written down because the pretty version of
this story would have been easy to believe and it is false.

## The six configurations, ranked (AUC on the held-out region)

| configuration | w016 | 0814 | 1667 | mean over segments |
|---|---|---|---|---|
| averaged prediction + TTA | 0.8974 | 0.8646 | **0.9432** | **0.9017** |
| averaged prediction | 0.9044 | 0.8676 | 0.9237 | 0.8986 |
| seed43 + TTA | 0.8911 | 0.8624 | 0.9396 | 0.8977 |
| seed43 alone | **0.9087** | 0.8433 | 0.9197 | 0.8906 |
| seed42 + TTA | 0.8874 | 0.8510 | 0.9244 | 0.8876 |
| seed42 alone | 0.8014 | **0.8683** | 0.8717 | 0.8471 |

No configuration wins on all three segments: the best one for w016 is fourth on the list and the best
one for 0814 is last. But on average the order is clear, and the distance between drawing a single
seed with bad luck (0.8471) and averaging the two (0.8986) is **0.05 of AUC without training
anything**.

With three segments it cannot be claimed that adding TTA on top of the averaged prediction is better
than the averaged prediction alone: 0.003 separates them and the sign changes depending on the
segment.

## What this does NOT say

It does not say the models are badly trained. An AUC of 0.85 on unseen data, in a problem where the
de facto standard is still "a person looked at it", is a good result. What is being corrected is the
reading of the published number, not the work.

Nor does it say which of the two seeds is better: with three segments there is nothing to state it
with.
