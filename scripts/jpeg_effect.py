"""Does JPEG compression of the published ds8 rasters change the legibility score?

The atlas scores connected components after a per-image Otsu binarisation. JPEG is lossy at
the 8x8 block scale, which is exactly the scale at which the binarisation decides whether two
blobs touch. So the whole atlas could be measuring compression artefacts.

Three renderings of the same prediction, to separate two effects that are otherwise confounded:

  jpg     the published ds8 JPEG - what the atlas actually scored
  area    the full-res TIFF box-averaged by 8 ourselves - no loss at all
  requant area, JPEG round-tripped at the published quality - isolates the loss with the
          resampling held fixed

area vs requant is the answer to the question. jpg vs area also folds in whatever resampling
the publisher used, which we do not know.
"""

import csv
import os
import sys

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 40))

import cv2
import numpy as np
from PIL import Image

import boto3
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_atlas import DS, image_threshold, score_window

BUCKET = "vesuvius-challenge-open-data"
WINDOW_UM, STRIDE_UM = 19652.0, 9826.0
BIG, HUGE, CAP = 960.0, 1920.0, 7200.0
MIN_COV = 0.7
THRESH_CAL = 0.900

TARGETS = [
    ("PHercParis4", "20260603145540-5753_-4", "canon", 2.4,
     "PHercParis4/segments/20260603145540-5753_-4/ink-detection/PHercParis4-20260603145540-2.4um-0.22m-78keV-volume-20260411134726-20260417190342-new_canon_autoresearch_recipe-tile256-stride128"),
    ("PHerc0172", "20251211135303-w054", "other", 7.91,
     "PHerc0172/segments/20251211135303-w054_20251211135303394_flatboi/ink-detection/PHerc0172-20251211135303-7.91um-53keV-volume-20241024131838-20250713185324-timesformer_scroll5_july_retreat-tile64-stride16"),
    ("PHercParis4", "20231031143852", "canon", 2.4,
     "PHercParis4/segments/20231031143852/ink-detection/PHercParis4-20231031143852-2.4um-0.22m-78keV-volume-20260411134726-20260417190342-new_canon_autoresearch_recipe-tile256-stride128"),
    ("PHercParis4", "20231031143852", "1um_s1z2", 2.258,
     "PHercParis4/segments/20231031143852/ink-detection/PHercParis4-20231031143852-1.129um-0.23m-78keV-volume-20260608103018-L1-20260709123958-mrg20736-1um-s1z2-tile256-stride128"),
    ("PHerc0139", "20250108000000-w025", "canon", 2.399,
     "PHerc0139/segments/20250108000000-w025_2025010863/ink-detection/PHerc0139-20250108000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128"),
    ("PHerc1667", "20240304161941-w023", "canon", 2.399,
     "PHerc1667/segments/20240304161941-w023_20240304161941_flatboi/ink-detection/PHerc1667-20240304161941-2.399um-0.22m-78keV-volume-20251217075048-20260417190342-new_canon_autoresearch_recipe-tile256-stride128"),
]

CACHE = "data/jpeg_effect"


def fetch(s3, key):
    local = os.path.join(CACHE, key.replace("/", "_"))
    if not os.path.exists(local):
        s3.download_file(BUCKET, key, local)
    return local


def jpeg_quality(path):
    """Recover the encoder quality from the quantisation table, so the round trip matches."""
    q = Image.open(path).quantization
    base = np.array([16, 11, 10, 16, 24, 40, 51, 61], dtype=float)
    got = np.array(q[0][:8], dtype=float)
    scale = np.median(got / base) * 100.0
    q = (200.0 - scale) / 2.0 if scale < 100 else 5000.0 / scale
    return int(min(100, max(1, round(q))))


def windows(img, um):
    wpx = max(int(round(WINDOW_UM / (um * DS))), 32)
    spx = max(int(round(STRIDE_UM / (um * DS))), 16)
    h, w = img.shape
    for y in range(0, max(h - wpx, 0) + 1, spx):
        for x in range(0, max(w - wpx, 0) + 1, spx):
            if img[y:y + wpx, x:x + wpx].shape[0] >= wpx // 2:
                yield y, x, wpx


def score_all(img, um):
    t = image_threshold(img)
    out = {}
    for y, x, wpx in windows(img, um):
        win = img[y:y + wpx, x:x + wpx]
        if win.shape[1] < wpx // 2:
            continue
        if float((win > 0).mean()) < MIN_COV:
            continue
        out[(y, x)] = score_window(win, t, um, BIG, HUGE, CAP)["mass_letter"]
    return out, t


def main():
    os.makedirs(CACHE, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    rows = []

    for scroll, seg, recipe, um, stem in TARGETS:
        jkey = stem.replace("/ink-detection/", "/ink-detection/downsampled/") + "-ds8.jpg"
        tkey = stem + ".tif"
        print(f"\n=== {scroll} {seg} {recipe} pitch={um} ===")
        jpath, tpath = fetch(s3, jkey), fetch(s3, tkey)

        jpg = cv2.imread(jpath, cv2.IMREAD_GRAYSCALE)
        full = cv2.imread(tpath, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if full is None:
            print("  unreadable tif"); continue
        if full.dtype != np.uint8:
            print(f"  tif dtype={full.dtype} range={full.min()}-{full.max()} -> scaling to 8bit")
            full = cv2.normalize(full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        area = cv2.resize(full, (full.shape[1] // DS, full.shape[0] // DS),
                          interpolation=cv2.INTER_AREA)
        q = jpeg_quality(jpath)
        _, buf = cv2.imencode(".jpg", area, [cv2.IMWRITE_JPEG_QUALITY, q])
        requant = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        _, buf50 = cv2.imencode(".jpg", area, [cv2.IMWRITE_JPEG_QUALITY, 50])
        stress = cv2.imdecode(buf50, cv2.IMREAD_GRAYSCALE)

        print(f"  full={full.shape} jpg={jpg.shape} area={area.shape} q~{q}")
        if abs(jpg.shape[0] - area.shape[0]) > 2 or abs(jpg.shape[1] - area.shape[1]) > 2:
            print("  SHAPE MISMATCH - skipping")
            continue
        crop = (min(jpg.shape[0], area.shape[0]), min(jpg.shape[1], area.shape[1]))
        jpg, area, requant, stress = (a[:crop[0], :crop[1]]
                                      for a in (jpg, area, requant, stress))

        s_j, t_j = score_all(jpg, um)
        s_a, t_a = score_all(area, um)
        s_r, t_r = score_all(requant, um)
        s_s, _ = score_all(stress, um)
        keys = sorted(set(s_j) & set(s_a) & set(s_r) & set(s_s))
        if not keys:
            print("  no shared windows"); continue
        vj = np.array([s_j[k] for k in keys])
        va = np.array([s_a[k] for k in keys])
        vr = np.array([s_r[k] for k in keys])
        vs = np.array([s_s[k] for k in keys])

        print(f"  otsu: jpg={t_j} area={t_a} requant={t_r}   windows={len(keys)}  q={q}")
        print(f"  mean mass_letter   jpg={vj.mean():.4f}  area={va.mean():.4f}  "
              f"requant={vr.mean():.4f}  q50={vs.mean():.4f}")
        print(f"  JPEG LOSS ONLY (requant-area): mean d={np.mean(vr - va):+.4f} "
              f"max|d|={np.abs(vr - va).max():.4f} r={np.corrcoef(va, vr)[0, 1]:.4f}")
        print(f"  published vs lossless (jpg-area): mean d={np.mean(vj - va):+.4f} "
              f"max|d|={np.abs(vj - va).max():.4f} r={np.corrcoef(va, vj)[0, 1]:.4f}")
        print(f"  STRESS q50 (q50-area): mean d={np.mean(vs - va):+.4f} "
              f"max|d|={np.abs(vs - va).max():.4f} r={np.corrcoef(va, vs)[0, 1]:.4f}")
        print(f"  windows >= {THRESH_CAL}:  jpg={int((vj >= THRESH_CAL).sum())}  "
              f"area={int((va >= THRESH_CAL).sum())}  requant={int((vr >= THRESH_CAL).sum())}"
              f"  q50={int((vs >= THRESH_CAL).sum())}")
        for k, j, a, r, st in zip(keys, vj, va, vr, vs):
            rows.append({"scroll": scroll, "segment": seg, "recipe": recipe, "pitch_um": um,
                         "y0": k[0] * DS, "x0": k[1] * DS, "jpg": round(float(j), 4),
                         "area": round(float(a), 4), "requant": round(float(r), 4),
                         "q50": round(float(st), 4)})

    if rows:
        with open("data/jpeg_effect.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        j = np.array([r["jpg"] for r in rows]); a = np.array([r["area"] for r in rows])
        rq = np.array([r["requant"] for r in rows]); q5 = np.array([r["q50"] for r in rows])
        print(f"\n=== POOLED ({len(rows)} windows, {len(TARGETS)} predictions) ===")
        for name, v in (("JPEG LOSS ONLY", rq), ("PUBLISHED vs LOSSLESS", j),
                        ("STRESS q50", q5)):
            print(f"{name:<24} mean d={np.mean(v - a):+.4f}  sd={np.std(v - a):.4f}  "
                  f"r={np.corrcoef(a, v)[0, 1]:.4f}  flips@0.900="
                  f"{int(((v >= THRESH_CAL) != (a >= THRESH_CAL)).sum())}")
        print("wrote data/jpeg_effect.csv")


if __name__ == "__main__":
    main()
