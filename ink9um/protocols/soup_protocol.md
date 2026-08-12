# Weight soup on the held-out region (pre-registered 2026-08-10)

Written BEFORE building any soup and BEFORE running any inference. Data seen up to this point:
`../results/holdout9_results.md`, `../results/step_sweep_results.md`,
`../results/zwindow_results.md`.

## What it is and why here

Averaging **predictions** (two models, two inferences, mean of the outputs) is already measured: it
gains up to 0.103 over the worse seed and loses almost nothing against the better one. A **weight
soup** averages the weight files number by number and leaves **a single model**, so it tries to
capture that same benefit while paying for one inference instead of two.

Two reasons to try it right here, both of them coming out of our own measurements:

1. The fact that averaging outputs works so well means the models **make their mistakes in different
   places**. That is where an averaging has something to gain.
2. The step sweep measured that at the end of training the checkpoints of the two seeds **diverge**
   on unseen papyrus (mean separation 0.0145 at 20k against 0.0601 at 75k) while they converge on the
   supervised region. Averaging along the training trajectory is the classic remedy for exactly that
   final instability.

## Verified structure of the checkpoints

Checked before writing this, not assumed: each `.pth` is `{model, config, step}`; the `model` has
**508 tensors, all float32**, with no integer counters or buffers that need separate handling; the
keys of seed42 and seed43 **match exactly**. The averaging is an arithmetic mean tensor by tensor,
with no special cases.

## The four soups, fixed NOW

**Exactly these four and no others** are evaluated. With 14 checkpoints there are thousands of
possible subsets and picking the best after seeing them would be manufacturing the result.

| name | what it averages | how many models come out |
|---|---|---|
| **A4** | the last 4 steps (40k, 50k, 60k, 75k) of one seed | 2 (one per seed) |
| **A7** | the 7 steps (10k to 75k) of one seed | 2 (one per seed) |
| **B** | seed42 and seed43, both at 75k | 1 |
| **AB** | all 14 checkpoints at once | 1 |

A4 and A7 are soups **within the same trajectory**, which is the case where the technique is well
established. B and AB cross seeds, which is the risky case.

## What it is compared against, by cost level

Comparing a soup against the average of two predictions would be cheating: they do not cost the same.
Two separate comparisons are declared:

- **Cost of ONE inference.** Baseline: `step-075000` of one seed, default z window, mean of the two
  seeds = **0.8558** on `pherc0814-46527`. A4, A7, B and AB compete.
- **Cost of TWO inferences.** Baseline: average of the predictions of the two seeds at 75k =
  **0.8676** on `pherc0814-46527`. The average of the predictions of the two A4 soups competes (and,
  if appropriate, A7).

## Criterion declared in advance

- **Signal**: a soup beats its baseline at the same cost level by **more than 0.01** of held-out AUC
  on `pherc0814-46527`. Same margin as the two previous sweeps.
- **No signal**: if none reaches 0.01, it is declared that the weight soup adds nothing on these
  checkpoints and that gets published as such, as was done with step 20k and with window S3.

Stage 2, only if there is signal: confirm the winning soup on `pherc0139-w016` and
`pherc1667-w029`, **without re-selecting**. If it does not hold up on both, it is reported as not
replicated.

## Mandatory control: telling "does not help" apart from "is broken"

The AUC on the **training region** is measured for every soup. The single checkpoints give 0.999
there.

- If a soup keeps ~0.999 on training and does not improve the held-out one: the soup **works as a
  model** and simply adds nothing. That is a result about the technique.
- If a soup collapses on training as well (declared threshold: **below 0.99**): the soup is
  **broken**, the averaged weights do not form a valid model. That says nothing about whether the
  technique would help; it says that these weights do not live in the same region. It gets reported
  as a broken soup and **not** as evidence against the technique.

This distinction is written down now because afterwards it would be convenient to confuse them.

## Prior diagnostic, cheap and with no GPU

Before spending inference, the **relative distance between weights** is measured: between the two
seeds at 75k, and between steps of the same seed. It is arithmetic over files that are already on
disk.

Declared prediction: if the two seeds started from different initializations, their weights will be
very far apart and **B and AB should come out broken**. This is written before looking at it so that
it counts as a prediction and not as an after-the-fact explanation.

This diagnostic **does not select** any soup: all four get run regardless of what comes out.

## What this experiment CANNOT say

It cannot say that weight soups are useless for ink detection models in general. It can only say
whether they are useful **on these 14 published checkpoints**, which is exactly what matters to
whoever downloads them.
