"""Execute the walkthrough notebook without a Jupyter install, and write its stdout back.

The notebook is the one artifact a reviewer is most likely to RUN rather than read, so a
stale or broken cell in it is worse than a stale paragraph anywhere else. This runs every
code cell in order in a single namespace - the same thing a kernel does - and refreshes the
text output of the cells that only print.

Image outputs are left exactly as they were: this script has no kernel and cannot produce
the display_data payload, and overwriting a figure with nothing would silently empty the
plot in the rendered notebook.
"""

import argparse
import contextlib
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", default="notebooks/atlas_walkthrough.ipynb")
    ap.add_argument("--write", action="store_true", help="write refreshed outputs back")
    args = ap.parse_args()

    path = os.path.abspath(args.nb)
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    # The notebook resolves its data paths from the working directory, so run it from where
    # a reader would: the notebooks/ folder.
    os.chdir(os.path.dirname(path))
    ns = {"__name__": "__main__"}
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        n += 1
        src = "".join(cell["source"])
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(src, f"<cell {n}>", "exec"), ns)
        except Exception:
            print(buf.getvalue())
            print(f"\n--- cell {n} FAILED ---", file=sys.stderr)
            raise
        out = buf.getvalue()
        print(f"===== cell {n} =====")
        print(out, end="")
        cell["execution_count"] = n
        has_image = any(o.get("output_type") != "stream" for o in cell.get("outputs", []))
        if args.write and not has_image:
            cell["outputs"] = ([{"output_type": "stream", "name": "stdout",
                                 "text": out.splitlines(keepends=True)}] if out else [])
            for o in cell["outputs"]:
                o.setdefault("name", "stdout")

    if args.write:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
