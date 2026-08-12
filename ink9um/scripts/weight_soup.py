"""Average the weights of several checkpoints into a single model (a weight soup).

    python scripts/weight_soup.py -o output.pth input1.pth input2.pth ...
    python scripts/weight_soup.py --diag input1.pth input2.pth ...

The output is a named flag on purpose. With a positional output the first argument is a 272 MB
file that gets written and the rest are files that get read, and getting the order wrong
overwrites a checkpoint with a soup silently. For the same reason an existing output is refused
unless `--force` is passed.

Averaging *predictions* costs one inference per model, forever. Averaging *weights* costs a
single inference, but it only works if the checkpoints live in the same region of weight space:
if they started from different initialisations, the mean is not a model, it is noise. That is
what `--diag` is for. It measures the relative distance between weights before spending any GPU.

The 9 um checkpoints are `{model, config, step}` with 508 float32 tensors and no integer
counters, so the averaging is a plain arithmetic mean with no special cases. This script checks
that rather than assuming it.

Accumulation is done in float64 and cast back to float32 on save. With 4 float32 summands
float32 would be enough, so this is about reproducibility, not correctness: in float64 the
result does not depend on the order the files are passed in.
"""
import argparse
import os

import torch


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d, d["model"]


def diagnose(paths):
    # Accumulators are float64, tensor by tensor. Summing 68 million products in float32 loses
    # so much precision that the cosine comes out above 1, which is impossible.
    base_d, base = load(paths[0])
    n = sum(v.numel() for v in base.values())
    sq_base = sum(v.double().pow(2).sum() for v in base.values())
    print(f"reference {paths[0]}   {len(base)} tensors   {n:,} parameters")
    for p in paths[1:]:
        _, sd = load(p)
        if list(sd.keys()) != list(base.keys()):
            raise SystemExit(f"different keys in {p}: averaging would be a silent bug")
        dot = sum((sd[k].double() * base[k].double()).sum() for k in base)
        sq = sum(sd[k].double().pow(2).sum() for k in base)
        sq_diff = sum((sd[k].double() - base[k].double()).pow(2).sum() for k in base)
        # relative distance: 0 is identical, ~1.41 is what two unrelated vectors give
        d_rel = (sq_diff.sqrt() / sq_base.sqrt()).item()
        cos = (dot / (sq.sqrt() * sq_base.sqrt())).item()
        print(f"{p}   relative distance {d_rel:.4f}   cosine {cos:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true",
                    help="only measure distance and cosine, write nothing")
    ap.add_argument("-o", "--output", help="destination .pth (required unless --diag)")
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing output")
    ap.add_argument("inputs", nargs="+")
    a = ap.parse_args()

    if a.diag:
        diagnose(a.inputs)
        return

    output, inputs = a.output, a.inputs
    if not output:
        raise SystemExit("missing -o/--output (or use --diag to only measure)")
    if len(inputs) < 2:
        raise SystemExit("a soup needs at least two checkpoints")
    if output in inputs:
        raise SystemExit(f"the output {output} is also an input: that would destroy a checkpoint")
    if os.path.exists(output) and not a.force:
        raise SystemExit(f"{output} already exists; pass --force to overwrite it")

    base_d, base = load(inputs[0])
    # Accumulate in float64 and cast to the original dtype on save, so the soup does not depend
    # on the order the files are passed in and anyone replicating gets the same bits.
    acc = {k: v.double() for k, v in base.items()}
    non_float = [k for k, v in base.items() if not v.is_floating_point()]
    if non_float:
        raise SystemExit(f"{len(non_float)} non-floating tensors; averaging them would be a bug")

    for p in inputs[1:]:
        _, sd = load(p)
        if list(sd.keys()) != list(base.keys()):
            raise SystemExit(f"different keys in {p}")
        for k in acc:
            if sd[k].shape != acc[k].shape:
                raise SystemExit(f"different shape for {k} ({p})")
            acc[k] += sd[k].double()

    for k in acc:
        acc[k] /= len(inputs)
        acc[k] = acc[k].to(base[k].dtype)

    base_d["model"] = acc
    base_d["step"] = -1  # the soup does not correspond to any training step
    torch.save(base_d, output)
    print(f"soup of {len(inputs)} checkpoints -> {output}")


if __name__ == "__main__":
    main()
