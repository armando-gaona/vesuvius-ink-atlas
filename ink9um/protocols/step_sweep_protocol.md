# Training step sweep on the held-out region (pre-registered 2026-08-10)

Written BEFORE running any inference of the sweep. The only data seen up to this point is that of
`../results/holdout9_results.md`, all of it at `step-075000`.

## Hypothesis

It is already measured that these checkpoints memorize: 0.999 AUC on the supervised region against
0.80 to 0.92 on the held-out region, in all three segments.

If that is progressive memorization, **training longer makes it worse**. In that case the AUC on the
held-out region should rise at first and **fall** towards the end, while the one on the supervised
region keeps rising or stays flat. `step-075000` is the last published one and the one everybody is
going to use by default.

If the hypothesis holds, there already exists a published checkpoint that is better than the final
one on unseen papyrus, and nobody can see it with the metric that is reported.

## What gets run

The 14 published checkpoints: two seeds (`hybrid_3d2d-seed42`, `hybrid_3d2d-seed43`) by seven steps
(10k, 20k, 30k, 40k, 50k, 60k, 75k).

Stage 1: only `pherc0814-46527`, which is the small segment (2130 x 3455) and costs ~1 min per
inference. Twelve new runs (the 75k ones are already done).

Stage 2, **only if stage 1 gives signal**: confirm on `pherc0139-w016` and `pherc1667-w029`, only at
the winning step and at 75k, so as not to spend two hours of GPU on noise.

Metric: AUC on the held-out region, plane Z=10, exactly as in `scripts/holdout_9um.py`. No TTA, so as
not to mix two effects.

## Criterion declared in advance

- **Signal**: there exists a step S < 75000 whose held-out AUC, averaged over the two seeds, beats
  the one at 75000 by **more than 0.01**. That margin is larger than the effect of seed averaging
  (0.003) and of the order of the change we already know matters.
- **No signal**: if the maximum falls at 75000, or if the improvement of any earlier step is 0.01 or
  less, the final checkpoint is declared a reasonable choice and that is said as such.
  A sweep that comes out flat also gets published.

## Mandatory control

The AUC on the **training region** is measured at every step and reported next to the held-out one.

The claim only holds if that control behaves as expected: rising or staying flat towards 75k. If it
also falls, then step 75k is worse at everything and there is nothing interesting to say about
memorization, only a bad final checkpoint.

## Trap to avoid

With seven steps and three segments it is easy to find a maximum by chance and count it as a finding.
Hence: the winner is fixed with stage 1 on a single segment, and stage 2 checks it on the other two
**without re-selecting**. If the stage 1 winner does not hold up on the other two, it is reported as
not replicated.
