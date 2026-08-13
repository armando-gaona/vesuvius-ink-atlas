# The soup against a better baseline than the last checkpoint (2026-08-13)

This document exists because of an objection. A Vesuvius team member, reading
`faint_signal_results.md`, replied that the audit was not enough to draw a conclusion and listed
what would be needed: more segments, larger regions within them, **more representative individual
checkpoints**, and cross-scroll generalization.

Three of those four cannot be done with public data, and the census that shows it is at the bottom
of this page. The third one can be done today, costs no GPU, and the answer is uncomfortable enough
that it is better found here than by a reader.

Everything below is recomputed from the TIFFs by `scripts/checkpoint_baseline.py`, not chained from
values already rounded for printing in the two documents it draws on. Literal output in
`checkpoint_baseline_run.txt`.

---

## The objection, stated precisely

`soup_results.md` compares soup A4 against `step-075000`. That is a fair baseline for one question:
*I already downloaded the released checkpoint, should I soup what I have?* It is not a fair baseline
for the question a reviewer asks: *is the soup better than a single checkpoint somebody could have
picked?*

The material to answer it was already on disk. `step_sweep_results.md` measured `step-020000` on the
same three segments, the same held-out region, the same plane, the same metric, and with the same
selection procedure: 20k and A4 were **both** chosen on `pherc0814-46527` and then applied unchanged
to the other two. Neither got to re-select. The comparison is symmetric in its procedure, which is
what makes it worth reporting at all.

The two tables were never put side by side. They are now.

## Gate

Before computing anything new, the script reproduces the published held-out AUCs of both the
baseline and the soup, and the whole stage 1 step sweep, from the files on disk. All twelve
soup/baseline cells and all fourteen sweep cells match to four decimals (one cell prints 0.8525
against a published 0.8526, inside the 5e-4 tolerance). Several soup builds sit on disk, float32 and
float64 accumulations of the same soup, so picking the wrong one would silently answer a different
question.

## Head to head

Held-out AUC, plane Z=10, no TTA. Per seed:

| segment | seed | 75k | A4 soup | step 20k | 20k minus soup |
|---|---|---|---|---|---|
| pherc0814-46527 | 42 | 0.8683 | 0.8692 | 0.8834 | +0.0142 |
| pherc0814-46527 | 43 | 0.8433 | 0.8630 | 0.8931 | +0.0301 |
| pherc0139-w016 | 42 | 0.8014 | 0.8525 | 0.9012 | +0.0487 |
| pherc0139-w016 | 43 | 0.9087 | 0.8960 | 0.8995 | +0.0035 |
| pherc1667-w029 | 42 | 0.8717 | 0.8966 | 0.8723 | -0.0244 |
| pherc1667-w029 | 43 | 0.9197 | 0.9297 | 0.9045 | -0.0252 |

Segment means:

| segment | 75k | A4 soup | step 20k | soup minus 75k | 20k minus 75k | **20k minus soup** |
|---|---|---|---|---|---|---|
| pherc0814-46527 | 0.8558 | 0.8661 | 0.8883 | +0.0103 | +0.0325 | **+0.0221** |
| pherc0139-w016 | 0.8550 | 0.8743 | 0.9004 | +0.0193 | +0.0454 | **+0.0261** |
| pherc1667-w029 | 0.8957 | 0.9131 | 0.8884 | +0.0174 | -0.0073 | **-0.0248** |
| **mean** | 0.8688 | 0.8845 | 0.8923 | +0.0157 | +0.0235 | **+0.0078** |

**A single early checkpoint beats the soup on 2 of the 3 segments and on 4 of the 6 individual
runs.** On the mean it is ahead by +0.0078, which is below the 0.01 margin this project declared in
advance for calling anything a win.

## How solid is that, once pixels stop being counted as independent

Block bootstrap of the same statistic, 1000 resamples, tiles drawn with replacement, both seeds
resampled on the same tiles. Delta is 20k minus soup, so positive favours the single checkpoint.

| segment | delta | 64 px | 128 px | 256 px | 512 px |
|---|---|---|---|---|---|
| pherc0814-46527 | +0.0221 | [-0.0059, +0.0569] | [-0.0113, +0.0682] | [-0.0141, +0.0622] | [-0.0071, +0.0670] |
| pherc0139-w016 | +0.0261 | [-0.0048, +0.0618] | [+0.0005, +0.0487] | [-0.0074, +0.0451] | [-0.0166, +0.0458] |
| pherc1667-w029 | -0.0248 | [-0.0531, +0.0016] | [-0.0596, +0.0032] | [-0.0548, +0.0062] | [-0.0531, -0.0001] |

P(delta > 0): 0.70 to 0.94 on `pherc0814-46527`, 0.93 to 0.98 on `pherc0139-w016`, and 0.025 to
0.059 on `pherc1667-w029`, which is the segment that favours the soup. Blocks available: 53/17/6/2,
66/23/10/6 and 127/42/14/10.

**Neither side wins.** Two segments lean to the single checkpoint, one leans to the soup, and only
two of the twelve cells exclude zero, one in each direction. By the same replication rule that this
project used to report step 20k as "not replicated", *"20k beats the soup"* is also **not
replicated**, at 2 of 3.

## The part that rescues the soup, and it is not a consolation prize

The obvious reading of the table above is that the soup is a bad idea. The measurement says
something narrower and more useful. Where does A4 sit inside the family it averages?

Full seven-step sweep, `pherc0814-46527`, the only segment where every step is on disk:

| step | seed42 | seed43 | mean | inside A4 |
|---|---|---|---|---|
| 010000 | 0.8760 | 0.8697 | 0.8728 | no |
| 020000 | 0.8834 | 0.8931 | 0.8883 | no |
| 030000 | 0.8476 | 0.8893 | 0.8685 | no |
| 040000 | 0.8352 | 0.8523 | 0.8438 | yes |
| 050000 | 0.8910 | 0.8427 | 0.8668 | yes |
| 060000 | 0.8511 | 0.8426 | 0.8469 | yes |
| 075000 | 0.8683 | 0.8433 | 0.8558 | yes |
| **A4 soup** | **0.8692** | **0.8630** | **0.8661** | |

Of the four checkpoints A4 actually contains, **1 of 4 beats it on seed42 and 0 of 4 on seed43**,
so it wins 7 of those 8 seed-and-step cells. Read the columns rather than the mean here: the one
that gets through, 50k on seed42, loses on seed43, and 30k does the mirror image. **The only two
steps that beat A4 on both seeds are 10k and 20k, and it contains neither.**

So the soup is not broken and it is not a trick. It does what an ensemble is supposed to do, it
beats 7 of the 8 cells of its own family. **What is badly chosen is the family.** "Average
the last four published steps" is sound advice conditional on only looking at the end of training,
and the end of training turns out to be the worst stretch for generalizing to unseen papyrus.

This also strengthens something already in `step_sweep_results.md` rather than contradicting it. The
same team member noted elsewhere that early checkpoints generalize better **across** scrolls while
late ones do better **within** the same scroll. Every held-out region used here is on the same
segment and the same scroll as the supervised one, which is the terrain that favours the late
checkpoints. The early ones win anyway, now on 4 of 6 runs instead of 2 of 3.

## What changes in the recommendation

`soup_results.md` tells a reader who will run a single inference to average the last four steps. That
advice is not wrong, but it was incomplete, and the incompleteness was derivable from two documents
we had already published:

- **If you will run one inference and you are choosing among the published checkpoints**, the choice
  of *which* checkpoint matters at least as much as whether you soup it. `step-020000` is worth
  +0.0235 of mean held-out AUC over `step-075000`; the soup is worth +0.0157.
- **Neither choice is established.** 20k beats the soup on 2 of 3 segments, the soup beats 20k on
  the third, and n = 3 with these interval widths cannot separate them.
- **The two are not exclusive and this has not been measured.** Nothing here tests a soup built
  around the early steps. It is deliberately not run: this document audits the published claim, it
  does not go looking for a new configuration that wins, which would be selection on the same three
  segments that are already carrying every other result in this repository.

What survives untouched from `soup_results.md`: the soup replicates 3 of 3 **against the released
final checkpoint**, it cuts the seed lottery by more than half on all three segments, and it costs
nothing at inference time. Those statements did not depend on 20k and are unaffected.

## Why the other three requests cannot be answered here

Direct census of the bucket, 2026-08-13, probing for `_validation_mask.zarr` on every segment of
`ink_9um/labels/aligned-scrollprizeorg-21slices`:

| | segments |
|---|---|
| with `inklabels` | **24** |
| with `_validation_mask.zarr` | **3** (`pherc0139-w016`, `pherc0814-46527`, `pherc1667-w029`) |
| on the other scroll (`phercparis4-*`) | 8, **none** with a validation mask |

This is measured against the bucket rather than repeated from our own notes, which is the point: the
claim "three segments is the ceiling with public data" had been asserted in this repository from
memory, and now it is sourced.

- **More segments** and **larger regions within them** are the same wall. The held-out region is not
  a conservative crop that could be widened: it is exactly the annotated, non-supervised area, and
  outside it a correct detection would be scored as a false positive because there is no annotation
  to compare against.
- **Cross-scroll generalization** needs an annotated scroll the checkpoints did not train on. All
  eight `phercparis4` segments carry annotation, none carries a held-out region, and per the
  `ink_9um` README the aligned set went into training.

The five natively annotated segments (28 planes, annotated at Z=14) are not a way around this: they
are the published annotated set itself, where AUC exceeds 0.995 because the model almost certainly
trained there.

## Limitations

| limitation | detail |
|---|---|
| n = 3 segments | sourced above, not assumed |
| 2 seeds | the two that were published |
| one comparison point | 20k against A4, not a search over steps; the sweep exists in full on one segment only |
| the full family is known for one segment | the "1 of 4 inside the soup" result is `pherc0814-46527` only |
| plane Z=10 only | the only annotated plane of the aligned set |
| no TTA | deliberately, so effects do not mix |
| nothing here is cross-scroll | the objection that motivated this document is only partly answered, and the unanswered part is a data problem, not a compute one |
