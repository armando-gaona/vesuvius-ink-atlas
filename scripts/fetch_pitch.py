"""Read the TRUE pixel pitch of every published prediction, from the zarr it was rendered from.

The micron figure in a prediction's filename is the resolution of the source SCAN. It is not
the pitch of the raster, because renders are frequently taken from a downsampled level of the
multiscale pyramid: the `1um_s1z2` predictions declare 1.129um in the name but their zarr
reports `source_group: 1` and a level-0 scale of 2.258um - exactly 2x coarser.

Trusting the filename made the atlas demand letters twice as large, in pixels, from those
predictions than they can physically have. That is the same class of error that once ranked
PHerc0172 last, and it is fatal to any comparison between recipes.

Each segment publishes `surface-volumes/<volume-tag>.zarr/.zattrs`, and the volume tag also
appears inside the prediction filename, so the two can be matched by substring. The .zattrs
carries `canvas_size` and the level-0 `coordinateTransformations.scale`, which is the pitch.
`canvas_size / 8` is then checked against the ds8 JPEG it should describe, so a wrong match
fails loudly instead of silently poisoning the atlas.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import cv2
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "vesuvius-challenge-open-data"
DS = 8


def make_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                            max_pool_connections=32))


def surface_volumes(s3, scroll, segment):
    prefix = f"{scroll}/segments/{segment}/surface-volumes/"
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return [c["Prefix"] for c in r.get("CommonPrefixes", [])]


def read_attrs(s3, zarr_prefix):
    key = zarr_prefix + ".zattrs"
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    a = json.loads(body)
    ds0 = a["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    return {
        "canvas_w": a["canvas_size"][0],
        "canvas_h": a["canvas_size"][1],
        "pitch_um": float(ds0[-1]),
        "source_group": a.get("source_group", ""),
        "num_slices": a.get("num_slices", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    with open(args.index_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"predictions to resolve: {len(rows)}")

    segs = sorted({(r["scroll"], r["segment"]) for r in rows})
    s3 = make_client()

    vols = {}

    def load_seg(sk):
        try:
            return sk, surface_volumes(s3, *sk)
        except Exception as e:
            return sk, []

    with ThreadPoolExecutor(args.workers) as ex:
        for i, (sk, v) in enumerate(ex.map(load_seg, segs), 1):
            vols[sk] = v
            if i % 40 == 0:
                print(f"  listed {i}/{len(segs)} segments")

    attrs_cache = {}

    def resolve(r):
        name = os.path.basename(r["key"])
        cands = vols.get((r["scroll"], r["segment"]), [])
        hit = None
        for p in cands:
            tag = p.rstrip("/").split("/")[-1][:-len(".zarr")]
            if tag in name:
                hit = p
                break
        if hit is None:
            return {**r, "pitch_um": "", "status": "no-matching-zarr"}
        if hit not in attrs_cache:
            try:
                attrs_cache[hit] = read_attrs(s3, hit)
            except Exception as e:
                attrs_cache[hit] = {"err": str(e)[:60]}
        a = attrs_cache[hit]
        if "err" in a:
            return {**r, "pitch_um": "", "status": "attrs-" + a["err"]}

        img = cv2.imread(os.path.join(args.cache_dir, r["key"].replace("/", "_")),
                         cv2.IMREAD_GRAYSCALE)
        status = "ok"
        if img is not None:
            # The pitch is only meaningful if this zarr really is the raster behind this
            # JPEG. A 2% tolerance covers the rounding of an integer ds8 division.
            exp_w, exp_h = a["canvas_w"] / DS, a["canvas_h"] / DS
            if abs(img.shape[1] - exp_w) / exp_w > 0.02 or \
               abs(img.shape[0] - exp_h) / exp_h > 0.02:
                status = f"size-mismatch img={img.shape[1]}x{img.shape[0]} " \
                         f"zarr/8={exp_w:.0f}x{exp_h:.0f}"
        else:
            status = "no-local-jpeg"
        return {**r, "pitch_um": a["pitch_um"], "source_group": a["source_group"],
                "num_slices": a["num_slices"], "status": status}

    out = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(resolve, rows), 1):
            out.append(res)
            if i % 50 == 0:
                print(f"  resolved {i}/{len(rows)}")

    fields = list(rows[0].keys()) + ["pitch_um", "source_group", "num_slices", "status"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(out)
    print(f"wrote {args.out_csv}")

    bad = [r for r in out if r["status"] != "ok"]
    print(f"\nresolved ok: {len(out) - len(bad)} / {len(out)}")
    for r in bad[:15]:
        print(f"  {r['status']:<50} {os.path.basename(r['key'])[:70]}")

    print(f"\n{'declared um':>12}{'true pitch':>12}{'group':>7}{'n':>7}")
    tab = {}
    for r in out:
        if r["pitch_um"] == "":
            continue
        k = (r["resolution_um"], r["pitch_um"], r["source_group"])
        tab[k] = tab.get(k, 0) + 1
    for (d, p, g), n in sorted(tab.items(), key=lambda kv: -kv[1]):
        print(f"{d:>12}{p:>12}{str(g):>7}{n:>7}")


if __name__ == "__main__":
    main()
