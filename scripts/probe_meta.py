"""Find the authoritative pixel pitch of a rendered segment.

The filename's `um` is the source scan, not the raster pitch - proven by paired segments
having identical pixel dimensions under scans that differ 2x. So the pitch has to come from
metadata, not from the name. This dumps whatever the bucket publishes alongside a segment.
"""

import argparse
import json

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "vesuvius-challenge-open-data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--get", default="", help="Key suffix to fetch and print")
    args = ap.parse_args()

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    if args.get:
        body = s3.get_object(Bucket=BUCKET, Key=args.get)["Body"].read()
        try:
            print(json.dumps(json.loads(body), indent=2)[:4000])
        except Exception:
            print(body[:3000].decode("utf-8", "replace"))
        return

    def walk(prefix, depth):
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
        for o in r.get("Contents", []):
            print(f"  FILE {o['Key']}  {o['Size']}")
        for c in r.get("CommonPrefixes", []):
            print(f"DIR  {c['Prefix']}")
            if depth > 1:
                walk(c["Prefix"], depth - 1)

    walk(args.prefix, args.depth)


if __name__ == "__main__":
    main()
