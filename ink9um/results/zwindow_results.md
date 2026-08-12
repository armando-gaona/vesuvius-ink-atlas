# Z window sweep (2026-08-10)

Protocol pre-registered in `../protocols/zwindow_protocol.md`. Stage 1 on segment
`pherc0814-46527`, `step-075000`, both seeds, no TTA. AUC on the held-out region, plane Z=10.
Aggregates recomputed in one pass with `scripts/recompute_zwindow_tables.py`.

---

## The official README makes two suggestions. Both were tested.

The `ink_9um` README says, verbatim:

> "If a checkpoint is not responding well on your data, it might just be a z layer offset; the
> models can be quite sensitive to it, and picking a different z window (`--layer-start`/
> `--layer-end`) can help. Averaging predictions over a few nearby z windows also works as a simple
> ensemble."

That is two separate pieces of advice, and they do not fare the same way.

| the suggestion | what was measured | verdict |
|---|---|---|
| shifting the z window can help | the best position gains +0.0113 on the segment where it was chosen, but +0.0030 and -0.0011 on the other two | **correct as written, and only as written.** It helps *per segment*. There is no globally better window to recommend |
| averaging a few nearby windows works as a simple ensemble | +0.0010 for three windows, +0.0003 for five | **buys nothing measurable here**, at three to five times the inference cost |

The second one is the useful negative result for anyone who downloads a checkpoint. The first one is
a confirmation, not a refutation: the README says "on your data" and "can help", and the data agree.
What does not survive is the stronger reading that a better default window exists.

---

## Stage 1: the five positions

The five are **all** the ones that fit (17 planes inside 21), not a hand-picked grid. S2 is the
default.

| position | indices | train. s42 | train. s43 | **held. s42** | **held. s43** | held. mean |
|---|---|---|---|---|---|---|
| S0 | 0 to 16 | 0.9998 | 0.9997 | 0.8224 | 0.8307 | 0.8266 |
| S1 | 1 to 17 | 0.9998 | 0.9997 | 0.8367 | 0.8377 | 0.8372 |
| **S2 (default)** | 2 to 18 | 0.9997 | 0.9997 | 0.8683 | 0.8433 | 0.8558 |
| **S3** | 3 to 19 | 0.9997 | 0.9997 | **0.8828** | **0.8514** | **0.8671** |
| S4 | 4 to 20 | 0.9998 | 0.9996 | 0.8802 | 0.8483 | 0.8642 |

### Mandatory control: the supervised region

Across the ten runs the AUC on the supervised region stays between 0.9996 and 0.9998, a total range
of 0.00016. Nothing here is a broken model: whatever the z window does, it does not damage the fit
on the region the model was trained on.

The same decision moves the held-out AUC far more. To avoid picking the friendlier number, all three
ranges:

| quantity | range across the five positions |
|---|---|
| held out, seed42 | 0.0603 |
| held out, seed43 | 0.0207 |
| held out, mean of the two seeds | **0.0405** |
| supervised, both seeds | 0.00016 |

The honest headline is the third row, **0.0405**, not the first. The 0.060 figure is one seed's
range and the other seed gives a third of that.

What this says is a fact about where a measurement is informative, not a claim about the authors:
**the supervised region cannot be used to choose a z window, because it barely moves.** The
held-out region can. That is the whole reason this package scores on `_validation_mask.zarr`.

---

## By the declared criterion: there is signal, and it is tight

S3 beats S2 by **+0.0113** on the mean of the two seeds, above the 0.01 margin fixed in advance.
Signal is declared on this segment.

Honestly, the margin is narrow and does not stand on its own:

- seed42 gains +0.0145 with S3, seed43 only +0.0081. **With seed43 alone the margin would not have
  been cleared.**

What is strong, and it is not the margin, is that **the ordering is identical in both seeds**:
S0 < S1 < S2 < S3, with S4 slightly below S3. Two independent seeds order five positions the same
way. The direction (a deeper window is better, with a ceiling at S3) is far more solid than the size
of the jump.

## The ensembles, in detail

The two ensembles were fixed in advance:

| configuration | held. s42 | held. s43 | mean | against S2 |
|---|---|---|---|---|
| S2 (default) | 0.8683 | 0.8433 | 0.8558 | |
| E3 (S1+S2+S3) | 0.8665 | 0.8471 | 0.8568 | +0.0010 |
| E5 (all five) | 0.8642 | 0.8479 | 0.8561 | +0.0003 |
| **S3 alone** | **0.8828** | **0.8514** | **0.8671** | **+0.0113** |

Window averaging buys nothing here. It is understandable why: averaging drags in the shallow
positions, which are the bad ones. On this segment, picking the window well gains about ten times
more than averaging several, at a third of the cost.

## How it degrades, the same signature again

Mean of the prediction where there is NO ink, held-out region, seed42: 0.3123 on S0 and 0.2937 on
S4. The mean where there IS ink barely moves (0.544 to 0.572). So the better window does not make
the model hit harder on the ink: it makes it **stop hesitating about the background**.

---

## Stage 2: the winner does NOT replicate

S3 was run on the other two segments without re-selecting the position. S2 for those two was already
measured in `holdout9_results.md`.

| segment | held. mean S2 | held. mean S3 | delta |
|---|---|---|---|
| pherc0814-46527 | 0.8558 | 0.8671 | **+0.0113** |
| pherc0139-w016 | 0.8550 | 0.8580 | +0.0030 |
| pherc1667-w029 | 0.8957 | 0.8946 | **-0.0011** |

Only the segment where the choice was made clears the 0.01 margin. The other two give +0.003 and
-0.001, which is to say nothing. **By the rule written in advance this is reported as not
replicated.** It cannot be turned into "use window 3 to 19".

A caveat that is written down because it is real, not because it rescues anything: **5 of the 6
individual measurements go up with S3.**

| segment | seed42 | seed43 |
|---|---|---|
| pherc0814-46527 | +0.0145 | +0.0081 |
| pherc0139-w016 | +0.0022 | +0.0038 |
| pherc1667-w029 | +0.0058 | **-0.0081** |

A sign test with 5 of 6 gives p = 0.22: not significant. The direction is consistent, the size is
not enough for a recommendation.

This is exactly what the README itself describes: a **per-case z offset**, not a better window in
general.

## The two levers overlap

On `pherc0814-46527`, the gain of S3 measured on the average of the two seeds, which is the recipe
already being recommended:

| configuration | held. |
|---|---|
| seed average, S2 | 0.8676 |
| seed average, S3 | 0.8734 |

+0.0058, half of what it gains over a single seed (+0.0113). Averaging seeds already eats part of
what tuning the window would gain. The levers do not add up.

---

## Conclusion

**The default z window is a defensible choice.** No better one was found that holds across three
segments, and that is what is reported.

Two things survive this sweep and are worth carrying forward:

1. **Averaging predictions over nearby z windows does not pay for itself.** +0.0010 and +0.0003 for
   three and five times the inference cost. If you are going to spend more than one inference, spend
   it on the second seed instead: averaging the two seeds gives 0.8676 on this segment, against an
   expected 0.8558 for one seed picked at random, which is +0.0118 for twice the cost.
2. **A z window has to be chosen per segment or not at all**, and it cannot be chosen by looking at
   how the model does on the region it was trained on, because that number moves by 0.00016 while
   the held-out number moves by 0.0405. Anyone tuning this needs a held-out region, which on public
   data means one of the three segments that ship `_validation_mask.zarr`.

This is the third time the same shape has appeared, with three different levers and three
independent pre-registered protocols: seed, training step and z window. All three move the held-out
figure by between 0.02 and 0.10, and none of them moves the supervised figure past the fourth
decimal.
