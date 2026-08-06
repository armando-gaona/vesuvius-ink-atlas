"""Enumerate every published ink-detection prediction in the open bucket.

Builds the index the failure atlas is computed over. The ds8 JPEGs are ~3 MB each, so the
whole corpus is downloadable; the full-resolution TIFFs (250-300 MB each) are not, and the
atlas does not need them - legibility is a structure-at-letter-scale property and a letter
is still ~160 px wide at ds8.
"""

import argparse
import csv
import re

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "vesuvius-challenge-open-data"


def list_prefixes(s3, prefix):
    out = []
    token = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix, "Delimiter": "/"}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out += [p["Prefix"] for p in r.get("CommonPrefixes", [])]
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def list_objects(s3, prefix):
    out = []
    token = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    scrolls = [p.rstrip("/") for p in list_prefixes(s3, "")]
    print(f"top-level prefixes: {len(scrolls)}")

    rows = []
    for scroll in scrolls:
        segs = list_prefixes(s3, f"{scroll}/segments/")
        if not segs:
            continue
        n_with = 0
        for seg in segs:
            objs = list_objects(s3, f"{seg}ink-detection/downsampled/")
            for key, size in objs:
                if not key.endswith("-ds8.jpg"):
                    continue
                m = re.search(r"-(\d+\.?\d*)um-", key)
                rows.append({
                    "scroll": scroll,
                    "segment": seg.split("/")[-2],
                    "key": key,
                    "size_mb": round(size / 1e6, 2),
                    "resolution_um": m.group(1) if m else "",
                    "recipe": "canon" if "canon" in key else ("1um_s1z2" if "s1z2" in key else "other"),
                })
                n_with += 1
        print(f"{scroll:<24} segments={len(segs):>4}  predictions={n_with}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["scroll", "segment", "key", "size_mb",
                                           "resolution_um", "recipe"])
        wr.writeheader()
        wr.writerows(rows)

    total_mb = sum(r["size_mb"] for r in rows)
    print(f"\nTOTAL predictions: {len(rows)}  ({total_mb:.0f} MB of ds8 JPEGs)")
    print(f"distinct segments: {len({r['segment'] for r in rows})}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
