# Z window sweep on the held-out region (pre-registered 2026-08-10)

Written BEFORE running any inference of this sweep. Data seen up to this point:
`../results/holdout9_results.md` and `../results/step_sweep_results.md`, all of it at the default z
window.

## Where the question comes from

The official README (`data/ink9um_README.md`) says two things, both of them **without a single
figure**:

> "it might just be a z layer offset; the models can be quite sensitive to it, and picking a
> different z window (--layer-start/--layer-end) can help"

> "Averaging predictions over a few nearby z windows also works as a simple ensemble"

So the authors themselves point at the lever and recommend the averaging, but nobody has published
how much either of the two is worth on unseen papyrus. That is exactly the gap this test bench can
fill.

## How many windows exist, measured and not assumed

The model consumes **17 planes** (`in_chans=17`, from the checkpoint's `crop_size`) and the aligned
volume has **21**. With no arguments, `infer.py` takes `[0,21)` and crops the center, which gives
indices **2 to 18**, verified in the log of the baseline run (`layer indices=[2, 3, ..., 18]`).

With 17 out of 21 only **five** whole positions fit, and here they all are:

| position | `--layer-start` | `--layer-end` | indices | note |
|---|---|---|---|---|
| S0 | 0 | 17 | 0 to 16 | flush against the shallow side |
| S1 | 1 | 18 | 1 to 17 | |
| **S2** | 2 | 19 | 2 to 18 | **the default one, already measured** |
| S3 | 3 | 20 | 3 to 19 | |
| S4 | 4 | 21 | 4 to 20 | flush against the deep side |

This is not a sweep over a hand-picked grid: it is the complete space. That kills at the root any
suspicion of having kept trying until something came out.

Training jitters the window over 17 out of 21, so all five positions are positions the model **saw**
during training. This does not measure extrapolation, it measures whether any of the five is
systematically better on unseen papyrus.

## What gets run

Checkpoint: **`step-075000`, both seeds**. It is the one everybody downloads and the one the step
sweep left as the standing recommendation (20k did not replicate).

Stage 1: only `pherc0814-46527` (2130 x 3455, ~1 min per inference). Eight new runs: 4 positions x 2
seeds. S2 is already done.

Stage 2, **only if stage 1 gives signal**: confirm on `pherc0139-w016` and `pherc1667-w029`, only at
the winning position and at S2, without re-selecting.

Metric: AUC on the held-out region, plane Z=10, `scripts/holdout_9um.py`. No TTA, so as not to mix
two effects. Same `patch=128 stride=96 blend=gaussian` as the baseline run.

**The label plane is always Z=10 and it does not move.** What moves is what the model reads as input.
The comparison is still against the same ground truth.

## Criterion declared in advance

- **Signal**: there exists a position S other than S2 whose held-out AUC, averaged over the two
  seeds, beats S2 by **more than 0.01**. Same margin as the step sweep, for consistency and because
  it is larger than the effect of seed averaging (0.003).
- **No signal**: if the maximum falls at S2, or if the improvement of any other position is 0.01 or
  less, **the default centered window is declared the correct choice** and that is said as such.
  A flat sweep also gets published: it would be the first published figure on a lever the README
  describes as important.

## The two averagings, fixed NOW so they cannot be chosen afterwards

Besides the individual positions, exactly **two** ensembles are evaluated, decided here:

- **E3**: average of S1, S2, S3 (the three central ones).
- **E5**: average of all five.

No other subset will be evaluated. With five positions there are 26 possible subsets and picking the
best after seeing them would be manufacturing a result.

## Mandatory control

The AUC on the **training region** is measured at every position and reported alongside. The two
possible readings are written down now, before seeing the data, so that the convenient one cannot be
picked later:

- **If the good position is also the best one on training**: the z window is a lever of global
  quality. It is still a free and useful recipe, but anybody could have found it with the metric that
  is already published. It gets reported as an improvement, not as a finding.
- **If the training AUC stays flat (~0.999) while the held-out one moves**: the published metric is
  once again blind to a real difference, just like what happens with the seeds. That is what turns
  this into a finding and not a trick.

## Trap to avoid

Five positions by two seeds give ten numbers; the maximum of ten noisy numbers almost always looks
good. Hence: the winner is fixed with stage 1 on a single segment and stage 2 checks it on the other
two **without re-selecting**. If it does not hold up, it is reported as not replicated, exactly as
was done with step 20k.

## What this experiment CANNOT say

With three held-out segments it will not be possible to state which is the best window in general. At
most it will be possible to say whether the default one is defensible and how much it costs to get
the window wrong.
