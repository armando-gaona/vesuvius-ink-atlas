# Training step sweep on the held-out region (2026-08-10)

Protocol pre-registered in `../protocols/step_sweep_protocol.md`. The 14 checkpoints published in
`ink_9um/checkpoints` (two seeds by seven steps). AUC on the held-out region, plane Z=10,
without TTA.

## Stage 1: full sweep on pherc0814-46527

| step | train. seed42 | train. seed43 | **held. seed42** | **held. seed43** | held. mean |
|---|---|---|---|---|---|
| 10k | 0.9656 | 0.9533 | 0.8760 | 0.8697 | 0.8729 |
| 20k | 0.9903 | 0.9860 | 0.8834 | 0.8931 | **0.8883** |
| 30k | 0.9967 | 0.9955 | 0.8476 | 0.8893 | 0.8685 |
| 40k | 0.9987 | 0.9982 | 0.8352 | 0.8523 | 0.8438 |
| 50k | 0.9993 | 0.9986 | 0.8910 | 0.8427 | 0.8669 |
| 60k | 0.9996 | 0.9994 | 0.8511 | 0.8426 | 0.8469 |
| 75k | 0.9997 | 0.9997 | 0.8683 | 0.8433 | 0.8558 |

**The mandatory control passes**: the training AUC rises monotonically in both seeds, from
0.95 up to 0.9997, without a single dip. What is measured below is not a bad final checkpoint.

Stage 1 winner by the declared criterion: **20k**, with 0.8883 against 0.8558 of 75k
(+0.0325, above the 0.01 margin fixed in advance).

## Stage 2: the winner does NOT replicate cleanly

20k was run on the other two segments, without re-selecting the step.

| segment | held. mean 20k | held. mean 75k | delta |
|---|---|---|---|
| pherc0814-46527 | 0.8883 | 0.8558 | **+0.0325** |
| pherc0139-w016 | 0.9004 | 0.8550 | **+0.0454** |
| pherc1667-w029 | 0.8884 | 0.8957 | **-0.0073** |

Two out of three in favour, one against. **By the rule written in advance this is reported as not
replicated.** It cannot be said "use 20k".

Mean over the three: 0.892 at 20k against 0.869 at 75k. The direction is that one, but with three
segments and a counterexample it is not enough for a recommendation.

## What does hold up 3 out of 3: training longer separates the seeds

Distance between the two published seeds, measured as the absolute value of the AUC difference:

| segment | region | separation at 20k | separation at 75k |
|---|---|---|---|
| pherc0814-46527 | held-out | 0.0097 | 0.0250 |
| pherc0139-w016 | held-out | 0.0017 | **0.1073** |
| pherc1667-w029 | held-out | 0.0322 | 0.0481 |
| | **mean** | **0.0146** | **0.0601** |
| pherc0814-46527 | supervised | 0.0044 | 0.0000 |
| pherc0139-w016 | supervised | 0.0191 | 0.0002 |
| pherc1667-w029 | supervised | 0.0086 | 0.0003 |
| | **mean** | **0.0107** | **0.0002** |

The two directions are consistent across the three segments, and they are opposite:

- On the **supervised region**, training makes the two seeds **converge** until they are
  indistinguishable: mean separation from 0.0107 to 0.0002, that is fifty times less.
- On the **held-out region**, training makes them **diverge**: from 0.0146 to 0.0601, four times more.

**This is the finding.** At the end of training the two checkpoints are twins on the region they
were trained on and a lottery on the region they were not. On `pherc0139-w016` that lottery is worth
eleven points of AUC.

It is not that training too long breaks the model: the fit on the supervised region keeps improving
to the last step. It is that **the supervised region stops being able to tell the two checkpoints
apart at exactly the point where they become most different on new papyrus**, which is the final
stretch, which is where the checkpoint everybody downloads comes from. Anyone choosing between these
checkpoints needs a held-out region to choose with, and on public data that means one of the three
segments shipping `_validation_mask.zarr`.

A caveat on how strong this is: it is a comparison of two points, 20k against 75k. The direction is
consistent 3 of 3 in both regions, which is what makes it worth reporting, but the intermediate
steps on `pherc0814-46527` are not monotone. Taking the differences straight off the stage 1 table,
from 10k to 75k: 0.0063, 0.0097, 0.0417, 0.0171, 0.0483, 0.0085, 0.0250. What is measured is that
the endpoints differ, not that separation grows smoothly with training.

## Practical consequence

What `holdout9_results.md` says still holds: average the two seeds. And now it is understood why it
is needed, which before was just an empirical observation. The seed lottery is not bad luck, it is
what the final stretch of training produces.

AUC of the **averaged prediction** of the two seeds on the held-out region, by step. Note this is a
different quantity from the "held. mean" columns above, which are means of two AUCs; the two are not
interchangeable.

| segment | averaged prediction, 20k | averaged prediction, 75k |
|---|---|---|
| pherc0814-46527 | 0.8986 | 0.8676 |
| pherc0139-w016 | **0.9165** | 0.9044 |
| pherc1667-w029 | 0.9018 | 0.9237 |
| mean | 0.9056 | 0.8986 |

The average at 20k wins on two out of three and on average by 0.007, which is little. The choice
between 20k and 75k is not settled; the one between averaging and not averaging is.

## A hypothesis of mine that the measurement knocked down

I went looking for the classic overfitting curve: the held-out one rises and then falls. **It is not
there.** The held-out one has no shape, it bounces between 0.84 and 0.89 with no clean trend, and the
minimum falls at 40k, not at the end. What does have a shape is the **separation between seeds**,
which was not what I was looking for.

## What is missing to close it

Three segments are all that exist with a published `_validation_mask.zarr`, so n=3 is the ceiling
with public data. To state the divergence with confidence, either more seeds (only two are published)
or more held-out regions would be needed.
