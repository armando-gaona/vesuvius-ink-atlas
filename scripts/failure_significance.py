"""Is a prediction with zero legible windows failing, or is it just small?

The atlas shipped a list of predictions with no window over the calibrated threshold and called
it a failure list. That framing has a hole big enough to invalidate it: a prediction contributes
as few as a dozen independent windows, and on a scroll where 5% of windows are legible, twelve
windows come up empty 54% of the time with nothing wrong anywhere in the pipeline.

So "zero legible windows" is not evidence by itself. It becomes evidence only when the
prediction is large enough that zero is surprising.

The test is the simplest one that applies. Under the null "this prediction is an ordinary piece
of its own scroll", each independent window is legible with that scroll's own base rate p, so
P(zero | n) = (1-p)^n. Using the SCROLL's rate rather than the corpus rate matters: the corpus
rate is 68% PHercParis4, and testing a PHerc0172 prediction against it would flag the whole
scroll as anomalous by construction, which is assuming the conclusion.

Note what this test cannot do. It cannot say a prediction is fine - failing to reject a null on
twelve windows is not evidence of health, just absence of evidence of failure. And it deals
with size, not with the threshold: a prediction whose windows all sit just under 0.900 is
invisible here. Those are the ones the contact sheets are for.
"""

import argparse
import csv
import math
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # Clamped: floating point drift prints a "-0.00%" lower bound at k=0, and a negative
    # percentage in a published table is the kind of detail that makes a reader stop trusting
    # the rest of the numbers.
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-csv", default="data/atlas.csv")
    ap.add_argument("--failures-csv", default="data/failures.csv")
    ap.add_argument("--out-csv", default="data/failure_significance.csv")
    ap.add_argument("--legible-at", type=float, default=0.900)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    with open(args.atlas_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Only the independent lattice. The atlas strides half a window on both axes, so the full
    # grid covers every patch of papyrus about four times; treating those as four independent
    # trials would quadruple n and make every prediction look significant.
    lat = []
    for r in rows:
        step = int(r["window_px"]) // 2
        if (int(r["y0"]) // step) % 2 == 0 and (int(r["x0"]) // step) % 2 == 0:
            lat.append(r)

    per_scroll = defaultdict(lambda: [0, 0])
    per_key = defaultdict(lambda: [0, 0])
    for r in lat:
        leg = float(r["mass_letter"]) >= args.legible_at
        for d, k in ((per_scroll, r["scroll"]), (per_key, r["key"])):
            d[k][0] += leg
            d[k][1] += 1

    print(f"base rate per scroll on the independent lattice (>= {args.legible_at})")
    base = {}
    for s, (k, n) in sorted(per_scroll.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        base[s] = k / n if n else 0.0
        lo, hi = wilson(k, n)
        print(f"  {s:<13}{k:>6} / {n:<7}{base[s]:>8.2%}   Wilson95 [{lo:.2%}, {hi:.2%}]")

    scroll_of = {r["key"]: r["scroll"] for r in rows}
    with open(args.failures_csv, encoding="utf-8") as f:
        fails = [r for r in csv.DictReader(f) if int(r["n_text_all"]) == 0]

    out, sig = [], 0
    for r in fails:
        s = scroll_of[r["key"]]
        n = per_key[r["key"]][1]
        p = base[s]
        # P(zero legible windows | n independent windows at this scroll's own base rate)
        pv = (1 - p) ** n
        out.append({"key": r["key"], "scroll": s, "segment": r["segment"],
                    "recipe": r["recipe"], "n_windows": n, "scroll_base_rate": round(p, 5),
                    "p_zero_by_chance": round(pv, 5),
                    "significant": int(pv < args.alpha), "max_score": r["max_score"]})
        sig += pv < args.alpha

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(sorted(out, key=lambda d: d["p_zero_by_chance"]))

    ns = sorted(d["n_windows"] for d in out)
    print(f"\n{len(out)} predictions with nothing over {args.legible_at} anywhere")
    print(f"  median independent windows each: {ns[len(ns) // 2]}")
    print(f"  surprising at alpha={args.alpha}:  {sig}")
    print(f"  explained by size alone:          {len(out) - sig}")
    for s in sorted({d["scroll"] for d in out}):
        g = [d for d in out if d["scroll"] == s]
        print(f"    {s:<13}{len(g):>4} failures{sum(d['significant'] for d in g):>5} surprising")
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
