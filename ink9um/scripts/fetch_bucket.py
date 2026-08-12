"""Download a folder from the Hugging Face bucket that hosts `ink_9um`, in parallel.

    python scripts/fetch_bucket.py <bucket path> <destination dir> [--workers 16]

Examples:

    # every checkpoint of one seed (7 files, 138 MB each)
    python scripts/fetch_bucket.py ink_9um/checkpoints/hybrid_3d2d-seed42 ckpt/seed42

    # the labels of one segment (3 zarrs, a few hundred small files)
    python scripts/fetch_bucket.py \\
        ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527 \\
        labels/pherc0814-46527

Why this exists rather than a one-line `curl`:

1. `ink_9um` is an HF *bucket*, not a model or dataset repo, so `huggingface-cli download`
   and `snapshot_download` do not reach it. The two endpoints that do are the tree API and
   the resolve URL, and both are used here.
2. The listing is **paginated at 1000 entries** and hands back the next page in an HTTP
   `Link: ...; rel="next"` header. A label folder is a Zarr, so it is hundreds of tiny files
   and it does cross that limit. Ignoring the header silently gives you a partial download,
   which is the worst possible failure mode for a label set.
3. Those tiny files make the bottleneck latency, not bandwidth, so they are fetched
   concurrently.
4. IPv6. On a network whose IPv6 is broken, `huggingface.co` publishes an AAAA record and
   downloads stall at a few files per second or hang outright, while S3 (IPv4 only) behaves
   perfectly and hides the cause. Resolution is filtered to IPv4 below.

Resumable: a file whose size on disk already matches the listing is skipped, so relaunching
after an interruption costs nothing.
"""
import argparse
import json
import os
import re
import socket
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/"
RESOLVE = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only(*args, **kwargs):
    return [r for r in _orig_getaddrinfo(*args, **kwargs) if r[0] == socket.AF_INET]


def listing(prefix):
    """Every file under `prefix`, following the rel="next" pages to the end."""
    url, out = API + prefix.strip("/"), []
    while url:
        with urllib.request.urlopen(url) as r:
            out += [e for e in json.load(r) if e.get("type") == "file"]
            link = r.headers.get("Link", "")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return out


def fetch(entry, prefix, dest):
    rel = entry["path"][len(prefix.strip("/")):].lstrip("/")
    target = os.path.join(dest, *rel.split("/"))
    if os.path.exists(target) and os.path.getsize(target) == entry["size"]:
        return 0
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    tmp = target + ".part"
    with urllib.request.urlopen(RESOLVE + entry["path"]) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, target)  # a half-written file must never look complete
    return entry["size"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="path inside the bucket, e.g. ink_9um/checkpoints/hybrid_3d2d-seed42")
    ap.add_argument("dest", help="local directory; the prefix itself is stripped from the paths")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    socket.getaddrinfo = _ipv4_only
    files = listing(a.prefix)
    if not files:
        raise SystemExit(f"nothing under {a.prefix}: check the path against the bucket listing")
    total = sum(e["size"] for e in files)
    print(f"{len(files):,} files, {total / 2**20:,.1f} MiB under {a.prefix}")

    with ThreadPoolExecutor(a.workers) as pool:
        got = list(pool.map(lambda e: fetch(e, a.prefix, a.dest), files))
    n = sum(1 for g in got if g)
    print(f"done: {n:,} downloaded, {len(files) - n:,} already present -> {a.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
