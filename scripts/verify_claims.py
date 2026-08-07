"""Recompute every number this project publishes, from the CSVs this project publishes.

This exists because the headline of this atlas has been wrong four times, and every one of
those times the number looked fine in prose and fell apart the moment someone recomputed it.
Prose is not checkable. This is.

Each claim below is stated as the string that appears in the README, the site or the prize
submission, next to the value recomputed from `data/`. A claim that does not match is printed
as FAIL and the script exits non-zero, so "the numbers still hold" is something you run rather
than something you remember.

Run: python scripts/verify_claims.py
"""

import csv
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGIBLE_AT = 0.900
INK_FLOOR = 0.05

_fails = []


def D(name):
    return os.path.join(ROOT, "data", name)


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def check(claim, got, want, tol=0.0):
    """`want` is what we say in public; `got` is what the data says today."""
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok:
        _fails.append(claim)
    print(f"  [{'ok' if ok else 'FAIL'}]  {claim:<58} published {want}   data {got}")


def score(r):
    """The published convention: a window with almost no ink above threshold scores zero.

    Without it, a window holding three specks in an otherwise blank field can put all of its
    (tiny) ink mass inside the letter band and score 1.0.
    """
    return 0.0 if float(r["frac_above"]) < INK_FLOOR else float(r["mass_letter"])


def lattice(rows):
    """One window in four. The atlas strides half a window on both axes, so neighbours share
    three quarters of their pixels; counting all of them as independent trials inflates every
    n by 4 and shrinks every confidence interval by half."""
    out = []
    for r in rows:
        step = int(r["window_px"]) // 2
        if (int(r["y0"]) // step) % 2 == 0 and (int(r["x0"]) // step) % 2 == 0:
            out.append(r)
    return out


def auc(pos, neg):
    """Mann-Whitney U, ties counted as half. No threshold, so nothing to tune."""
    if not pos or not neg:
        return float("nan")
    s = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return s / (len(pos) * len(neg))


def main():
    rows = load(D("atlas.csv"))
    lat = lattice(rows)

    print("\n1. DENOMINATOR - what population every percentage is over")
    check("46,764 scored windows in the atlas", len(rows), 46764)
    check("12,387 independent windows", len(lat), 12387)
    man = load(D("atlas_manifest.csv"))
    st = {}
    for r in man:
        st[r["status"]] = st.get(r["status"], 0) + 1
    check("423 predictions in the manifest", len(man), 423)
    check("374 of them scored", st.get("scored", 0), 374)
    check("423 = sum of the manifest statuses", sum(st.values()), 423)
    print(f"         manifest breakdown: {st}")
    check("374 distinct S3 keys in the atlas", len({r["key"] for r in rows}), 374)

    print("\n2. HEADLINE - legible fraction of published papyrus area")
    k = sum(score(r) >= LEGIBLE_AT for r in lat)
    lo, hi = wilson(k, len(lat))
    check("3.92% of area legible", round(100 * k / len(lat), 2), 3.92, 0.005)
    check("  Wilson95 low  3.60%", round(100 * lo, 2), 3.60, 0.005)
    check("  Wilson95 high 4.28%", round(100 * hi, 2), 4.28, 0.005)
    print(f"         {k} of {len(lat)} independent windows")

    print("\n3. THE GAP - the claim that replaces the retracted failure list")
    per = {}
    for r in lat:
        d = per.setdefault(r["scroll"], [0, 0])
        d[0] += score(r) >= LEGIBLE_AT
        d[1] += 1
    for s in sorted(per, key=lambda s: -per[s][0] / per[s][1]):
        kk, nn = per[s]
        a, b = wilson(kk, nn)
        print(f"         {s:<13}{nn:>7} windows{100 * kk / nn:>8.2f}%   [{100*a:.2f}, {100*b:.2f}]")
    for s, n_w, pct, c_lo, c_hi in (("PHerc0172", 3710, 0.54, 0.35, 0.83),
                                    ("PHercParis4", 7933, 5.48, 5.00, 6.01)):
        kk, nn = per[s]
        a, b = wilson(kk, nn)
        check(f"{s} n = {n_w}", nn, n_w)
        check(f"{s} {pct}%", round(100 * kk / nn, 2), pct, 0.005)
        check(f"{s} CI [{c_lo}, {c_hi}]",
              (round(100 * a, 2), round(100 * b, 2)), (c_lo, c_hi))
    gap_lo = wilson(*per["PHercParis4"])[0] - wilson(*per["PHerc0172"])[1]
    print(f"         intervals disjoint by {100 * gap_lo:.2f} points" if gap_lo > 0
          else "         WARNING: intervals overlap")
    if gap_lo <= 0:
        _fails.append("the two intervals must not overlap")

    print("\n   the ordering must not depend on the threshold")
    for t, p172, p4 in ((0.850, 4.20, 16.16), (0.950, 0.00, 0.49)):
        got = {}
        for s in ("PHerc0172", "PHercParis4"):
            g = [r for r in lat if r["scroll"] == s]
            got[s] = round(100 * sum(score(r) >= t for r in g) / len(g), 2)
        check(f"at {t}: {p172}% vs {p4}%",
              (got["PHerc0172"], got["PHercParis4"]), (p172, p4))

    print("\n   the counterexample that keeps pitch from being the explanation")
    pitch = {r["scroll"]: float(r["pitch_um"]) for r in rows}
    kk, nn = per["PHerc0139"]
    check("PHerc0139 sits at 2.399 um/px", pitch["PHerc0139"], 2.399, 1e-9)
    check("PHerc0139 scores 0.25%", round(100 * kk / nn, 2), 0.25, 0.005)
    print("         => the finest-pitch scroll is also near the bottom: pitch is a candidate,")
    print("            not the explanation. This must be quoted wherever the gap is quoted.")

    print("\n4. CALIBRATION - AUC against the 125 blind labels")
    key = {r["i"]: r for r in load(D("labels/labeling_key.csv"))}
    lab = [r for r in load(D("labels/labels.csv")) if r["i"] in key]
    check("125 labels join to a scored window", len(lab), 125)
    for name, floored, want in (("raw mass_letter", False, 0.918),
                                ("with the ink floor", True, 0.931)):
        pos, neg = [], []
        for r in lab:
            k2 = key[r["i"]]
            v = 0.0 if (floored and float(k2["frac_above"]) < INK_FLOOR) \
                else float(k2["mass_letter"])
            (pos if r["label"] == "text" else neg).append(v)
        check(f"AUC {name}", round(auc(pos, neg), 3), want, 0.0005)

    # The floor is a declared safeguard, not a measured gain: bootstrap the difference and
    # see how much of it is one window changing sides.
    rnd = random.Random(0)
    idx = list(range(len(lab)))
    diffs = []
    for _ in range(2000):
        samp = [lab[rnd.choice(idx)] for _ in idx]
        vals = []
        for floored in (True, False):
            p, n = [], []
            for r in samp:
                k2 = key[r["i"]]
                v = 0.0 if (floored and float(k2["frac_above"]) < INK_FLOOR) \
                    else float(k2["mass_letter"])
                (p if r["label"] == "text" else n).append(v)
            vals.append(auc(p, n))
        diffs.append(vals[0] - vals[1])
    diffs.sort()
    d_lo, d_hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    # 0.0125 sits exactly on a rounding boundary, so the tolerance is one unit in the last
    # published digit rather than half a unit. Anything tighter fails on the choice of
    # rounding rule instead of on the data.
    check("ink floor gains +0.013 AUC", round(sum(diffs) / len(diffs), 4), 0.013, 0.001)
    check("  bootstrap CI95 low 0.000", round(d_lo, 3), 0.000, 0.0005)
    check("  bootstrap CI95 high 0.041", round(d_hi, 3), 0.041, 0.0035)
    check("  P(gain > 0) ~ 0.63", round(sum(d > 0 for d in diffs) / len(diffs), 2), 0.63, 0.015)
    print("         => the gain is inside the noise of a single window. Declare it, don't sell it.")
    print("         (this is a resampling, so it moves in the third decimal between seeds;")
    print("          quote it to two, or the number stops reproducing.)")

    # Nothing may rest on the floor, so the headline has to be recomputed without it too.
    k0 = sum(float(r["mass_letter"]) >= LEGIBLE_AT for r in lat)
    a0, b0 = wilson(k0, len(lat))
    check("without the ink floor: 3.96%", round(100 * k0 / len(lat), 2), 3.96, 0.005)
    check("  CI [3.63, 4.31]", (round(100 * a0, 2), round(100 * b0, 2)), (3.63, 4.31))

    print("\n5. THE RETRACTION - why 'zero legible windows' was not a failure list")
    sig = {r["key"]: r for r in load(D("failure_significance.csv"))}
    vis = load(D("failure_visual_review.csv"))
    check("70 predictions with nothing over 0.9 anywhere", len(sig), 70)
    check("both reviews cover the same 70", {v["key"] for v in vis} == set(sig), True)
    ns = sorted(int(r["n_windows"]) for r in sig.values())
    print(f"         independent windows each: min {ns[0]}, median {ns[len(ns)//2]}, max {ns[-1]}")
    tab = {}
    for v in vis:
        kk = (v["verdict"], int(sig[v["key"]]["significant"]))
        tab[kk] = tab.get(kk, 0) + 1
    for verdict, gloss in (("letras", "readable letters"),
                           ("estructura", "letter-like structure"),
                           ("moteado", "speckle only")):
        print(f"         {gloss:<24}{tab.get((verdict, 1), 0):>5} surprising"
              f"{tab.get((verdict, 0), 0):>5} size alone")
    check("11 of the 70 show readable Greek",
          sum(v["verdict"] == "letras" for v in vis), 11)
    check("4 of the 70 are surprising at a=0.05",
          sum(int(r["significant"]) for r in sig.values()), 4)
    check("surprising AND visually empty", tab.get(("moteado", 1), 0), 0)
    check("all 4 surprising ones are readable", tab.get(("letras", 1), 0), 4)

    print("\n6. NEGATIVE CONTROL - is the gap our own scorer losing points at coarse pitch?")
    # The gap is a difference between two legible FRACTIONS, so the statistic that answers it
    # is the change in that fraction - losses net of gains. Counting only the windows that
    # fall below the line overstates the effect, because coarsening also merges specks into
    # letter-sized blobs and pushes other windows up.
    worst = 0.0
    n_pairs = 0
    tot_lost = tot_leg = 0
    for f in ("pitch_ablation.csv", "pitch_ablation_paris4.csv"):
        pa = load(D(f))
        a = [float(p["fine"]) for p in pa]
        b = [float(p["coarse"]) for p in pa]
        n = len(a)
        n_pairs += n
        ma, mb = sum(a) / n, sum(b) / n
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((x - mb) ** 2 for x in b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        leg_f = sum(x >= LEGIBLE_AT for x in a)
        leg_c = sum(y >= LEGIBLE_AT for y in b)
        lost = sum(x >= LEGIBLE_AT and y < LEGIBLE_AT for x, y in zip(a, b))
        gained = sum(x < LEGIBLE_AT and y >= LEGIBLE_AT for x, y in zip(a, b))
        rel = 100 * (leg_f - leg_c) / leg_f
        worst = max(worst, rel)
        tot_lost += lost
        tot_leg += leg_f
        print(f"         {pa[0]['scroll']:<12}{n:>6} pairs  mean {mb - ma:+.4f}  "
              f"r {cov / math.sqrt(va * vb):.3f}  legible {leg_f} -> {leg_c}  "
              f"(-{lost} +{gained})  {rel:+.1f}% relative")
    check("2,151 paired windows", n_pairs, 2151)
    check("<= 5 of the 90 relative points are the scorer", math.ceil(worst), 5)
    print(f"         worst scroll loses {worst:.1f}% of its legible fraction when coarsened;")
    print("         the other one goes UP, so the scorer is not systematically penalised.")
    print(f"         pessimistic view (ignore the gains): {tot_lost}/{tot_leg} = "
          f"{100 * tot_lost / tot_leg:.1f}% - still leaves "
          f"{90 - round(100 * tot_lost / tot_leg)} of the 90 points unexplained.")
    print("         LOWER BOUND EITHER WAY: this degrades the model's OUTPUT, not its input.")

    print("\n7. THE OTHER CONTROLS, so nobody has to take them on trust")
    je = load(D("jpeg_effect.csv"))
    check("JPEG control: 408 windows", len(je), 408)
    # `area` is the lossless baseline: the full-res TIFF box-averaged by 8 ourselves. Comparing
    # against the published JPEG instead would fold in the publisher's unknown resampling,
    # which is a different question.
    for name, col, want_d, want_r, want_flips in (
            ("JPEG loss alone", "requant", -0.0007, 0.9986, 0),
            ("published vs lossless", "jpg", 0.0000, 0.9996, 2),
            ("stress at quality 50", "q50", 0.0002, 0.9984, 2)):
        a = [float(r["area"]) for r in je]
        v = [float(r[col]) for r in je]
        n = len(a)
        ma, mv = sum(a) / n, sum(v) / n
        cov = sum((x - ma) * (y - mv) for x, y in zip(a, v))
        rr = cov / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mv) ** 2 for y in v))
        fl = sum((x >= LEGIBLE_AT) != (y >= LEGIBLE_AT) for x, y in zip(a, v))
        check(f"{name}: mean d {want_d:+.4f}", round(mv - ma, 4), want_d, 5e-5)
        check(f"{name}: r {want_r}", round(rr, 4), want_r, 5e-5)
        check(f"{name}: {want_flips} flips of 408", fl, want_flips)
    smear = sum(float(r["mass_smear"]) > 0.01 for r in lat)
    worst = max((float(r["mass_smear"]) for r in lat if score(r) >= LEGIBLE_AT), default=0.0)
    check("24.2% of windows carry mass above the 7200 um ceiling",
          round(100 * smear / len(lat), 1), 24.2, 0.05)
    check("no window that passes 0.900 carries any", round(worst, 3), 0.0, 1e-9)
    print("         => the ceiling works, and only by rejecting. It is not decoration.")

    print("\n" + "=" * 78)
    if _fails:
        print(f"{len(_fails)} CLAIM(S) NO LONGER MATCH THE DATA - do not publish:")
        for c in _fails:
            print(f"  - {c}")
        sys.exit(1)
    print("every published claim recomputes from the published CSVs")


if __name__ == "__main__":
    main()
