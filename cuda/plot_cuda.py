#!/usr/bin/env python3
"""Plot CUDA LBM sweep results (block/grid, problem size, memory layout).

Reads a CSV written by the sweep scripts with header:
    nx,ny,block_x,block_y,layout,steps,wall_time_s,mlups

The plot type is auto-detected from what varies in the file:
  - both layouts present            -> SoA vs AoS comparison (MLUPS vs cells)
  - several block shapes, one size  -> MLUPS vs threads/block
  - several sizes, one layout       -> MLUPS vs cells, with the V100 roofline

Dependencies: numpy, matplotlib (no pandas).

Usage:
    python3 plot_cuda.py block_results_*.csv
    python3 plot_cuda.py size_results_*.csv  --bw 900 --bytes 288
    python3 plot_cuda.py layout_results_*.csv
"""

import argparse
import csv
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def read_rows(paths):
    rows = []
    for pat in paths:
        for path in glob.glob(pat):
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    rows.append(dict(
                        nx=int(r["nx"]), ny=int(r["ny"]),
                        bx=int(r["block_x"]), by=int(r["block_y"]),
                        layout=r["layout"].strip(),
                        mlups=float(r["mlups"]),
                        cells=int(r["nx"]) * int(r["ny"]),
                        tpb=int(r["block_x"]) * int(r["block_y"]),
                    ))
    return rows


def roofline_mlups(bw_gbs, bytes_per_cell):
    return bw_gbs * 1e9 / bytes_per_cell / 1e6


def plot_block(rows, args):
    rows = sorted(rows, key=lambda r: (r["tpb"], r["bx"]))
    x = [r["tpb"] for r in rows]
    y = [r["mlups"] for r in rows]
    labels = [f"{r['bx']}×{r['by']}" for r in rows]

    fig, ax = plt.subplots(figsize=(5.5, 4.0), layout="constrained")
    ax.plot(x, y, "o-", markeredgecolor="black", markeredgewidth=0.5)
    for xi, yi, lb in zip(x, y, labels):
        ax.annotate(lb, (xi, yi), fontsize=6, ha="center", va="bottom")

    roof = roofline_mlups(args.bw, args.bytes)
    ax.axhline(roof, color="red", ls="--", lw=1,
               label=f"roofline ≈ {roof:.0f} MLUPS")

    ax.set_xscale("log", base=2)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("threads per block (block_x × block_y)")
    ax.set_ylabel("MLUPS")
    ax.set_title("CUDA block/grid sweep "
                 f"({rows[0]['nx']}×{rows[0]['ny']}, {rows[0]['layout']})")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize=8)
    _save(fig, "cuda_block")


def plot_size(rows, args):
    rows = sorted(rows, key=lambda r: r["cells"])
    x = [r["cells"] for r in rows]
    y = [r["mlups"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.5, 4.0), layout="constrained")
    ax.plot(x, y, "o-", markeredgecolor="black", markeredgewidth=0.5,
            label="measured")
    roof = roofline_mlups(args.bw, args.bytes)
    ax.axhline(roof, color="red", ls="--", lw=1,
               label=f"V100 roofline ≈ {roof:.0f} MLUPS "
                     f"({args.bw:.0f} GB/s, {args.bytes:.0f} B/cell)")

    ax.set_xscale("log")
    ax.set_xlabel("lattice cells (nx · ny)")
    ax.set_ylabel("MLUPS")
    ax.set_ylim(0, roof * 1.1)
    ax.set_title(f"CUDA problem-size sweep ({rows[0]['layout']}, "
                 f"{rows[0]['bx']}×{rows[0]['by']})")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize=8)
    _save(fig, "cuda_size")


def plot_layout(rows, args):
    by_layout = defaultdict(list)
    for r in rows:
        by_layout[r["layout"]].append(r)

    fig, ax = plt.subplots(figsize=(5.5, 4.0), layout="constrained")
    for layout, marker in zip(sorted(by_layout), ["o", "s"]):
        g = sorted(by_layout[layout], key=lambda r: r["cells"])
        ax.plot([r["cells"] for r in g], [r["mlups"] for r in g],
                marker=marker, markeredgecolor="black", markeredgewidth=0.5,
                label=layout)

    roof = roofline_mlups(args.bw, args.bytes)
    ax.axhline(roof, color="red", ls="--", lw=1,
               label=f"roofline ≈ {roof:.0f} MLUPS")

    ax.set_xscale("log")
    ax.set_xlabel("lattice cells (nx · ny)")
    ax.set_ylabel("MLUPS")
    ax.set_title(f"CUDA memory layout: SoA vs AoS ({rows[0]['bx']}×{rows[0]['by']})")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize=8)
    _save(fig, "cuda_layout")


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fn = f"{stem}.{ext}"
        fig.savefig(fn, dpi=150)
        print(f"saved {fn}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="results CSV(s) or glob")
    ap.add_argument("--bw", type=float, default=900.0,
                    help="V100 HBM2 bandwidth in GB/s (default 900)")
    ap.add_argument("--bytes", type=float, default=288.0,
                    help="bytes moved per cell per step (default 288)")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        print("No rows found.")
        return

    layouts = {r["layout"] for r in rows}
    sizes = {r["cells"] for r in rows}
    blocks = {(r["bx"], r["by"]) for r in rows}

    # Print achieved fraction of roofline.
    roof = roofline_mlups(args.bw, args.bytes)
    best = max(r["mlups"] for r in rows)
    print(f"roofline ≈ {roof:.0f} MLUPS; best measured {best:.0f} "
          f"({100*best/roof:.0f}% of roofline)")

    if len(layouts) > 1:
        plot_layout(rows, args)
    elif len(blocks) > 1 and len(sizes) == 1:
        plot_block(rows, args)
    else:
        plot_size(rows, args)


if __name__ == "__main__":
    main()
