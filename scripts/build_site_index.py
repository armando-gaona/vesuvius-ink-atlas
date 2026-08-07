"""Build data/site_index.json — the payload behind the browsable table on index.html.

Joins the per-prediction summary with the resolved pixel pitch, and attaches a direct
HTTPS link to the official ds8 prediction on the open bucket so a row in the table is one
click away from the image it scores. Nothing here recomputes the atlas; it only reshapes
what is already in data/ into something a static page can fetch in one request.

    python scripts/build_site_index.py
"""
import csv, hashlib, json, os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(ROOT, "data", *p)

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
THRESH = 0.900


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    summary = load(D("segment_summary.csv"))
    # Joined on `key`, the S3 path. (scroll, segment, recipe) is not unique - the same
    # segment appears twice under one recipe label - and joining on it silently gave one
    # prediction's pitch to another's row.
    pitch = {r["key"]: r for r in load(D("predictions_pitch.csv"))}

    # Maps are a curated subset, named <rank>_<segment[:24]>_<recipe>_<hash6>.png by
    # segment_map.py, where the hash is of the same key. Both the segment stem and the
    # recipe can contain underscores, so rather than parse the name we rebuild it.
    maps = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "figures", "maps", "*.png"))}

    rows = []
    for r in summary:
        p = pitch.get(r["key"], {})
        seg = r["segment"]
        row = {
            "scroll": r["scroll"],
            "segment": seg,
            "recipe": r["recipe"],
            "windows": int(r["windows"]),
            "n_text": int(r["n_text"]),
            # A prediction whose coverage filter left no independent window is unrankable,
            # which is not the same as zero percent readable.
            "pct": round(float(r["pct_text"]), 4) if r["pct_text"] else None,
            "ci": [float(r["ci_lo"]), float(r["ci_hi"])] if r["pct_text"] else None,
            "mean": round(float(r["mean_score"]), 4) if r["mean_score"] else None,
            "max": round(float(r["max_score"]), 4),
            "best": [int(r["best_y0"]), int(r["best_x0"])],
            "win_px": int(r["window_px"]),
            "cov": float(r["max_coverage"]) if r["max_coverage"] else None,
            "pitch": float(p["pitch_um"]) if p.get("pitch_um") else None,
            # resolution_um is what the FILENAME claims; pitch_um is what the raster is.
            "named": float(p["resolution_um"]) if p.get("resolution_um") else None,
            "level": int(p["source_group"]) if p.get("source_group", "") != "" else None,
            "url": BUCKET + "/" + r["key"],
        }
        row["mislabelled"] = bool(
            row["pitch"] and row["named"] and abs(row["pitch"] - row["named"]) > 1e-6
        )
        h = hashlib.sha1(r["key"].encode()).hexdigest()[:6]
        for rank in ("top", "bot"):
            name = f"{rank}_{seg[:24]}_{r['recipe']}_{h}.png"
            if name in maps:
                row["map"] = "figures/maps/" + name
                break
        rows.append(row)

    rows.sort(key=lambda r: (-(r["pct"] or 0), -r["windows"]))
    out = {
        "threshold": THRESH,
        "ranked_min_windows": 20,
        "generated_from": "data/segment_summary.csv + data/predictions_pitch.csv",
        "rows": rows,
    }
    path = D("site_index.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(rows)} predictions -> {path} ({os.path.getsize(path)/1024:.0f} KB)")
    print(f"  with map      : {sum('map' in r for r in rows)}")
    print(f"  mislabelled   : {sum(r['mislabelled'] for r in rows)}")
    print(f"  missing pitch : {sum(r['pitch'] is None for r in rows)}")


if __name__ == "__main__":
    main()
