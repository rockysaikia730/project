#!/usr/bin/env python3
"""Plot MPI strong- and weak-scaling results for the LBM solver.

Ingests the CSV files written by strong_scaling.sbatch and
weak_scaling.sbatch (header: nx,ny,np,steps,wall_time_s,mlups) and produces
the figures used in the report / slides:

  - strong_speedup.{png,pdf} : speedup S(P)=T(1)/T(P) per problem size,
                               with the Amdahl prediction (serial fraction
                               from profiling ~ 0 -> ideal linear) and a
                               marker at the single-socket core count.
  - weak_efficiency.{png,pdf}: weak efficiency E(P)=T(1)/T(P) vs P with the
                               Gustafson ideal (flat at 1).
  - mlups_throughput.{png,pdf}: absolute MLUPS vs P (handy for discussion /
                               the free-choice slide).

It also prints the efficiency tables to stdout.

Dependencies: numpy, matplotlib (no pandas).

Usage:
    python3 plot_scaling.py \
        --strong strong_results_*.csv \
        --weak   weak_results_*.csv \
        --cores-per-socket 36
"""

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter

rcParams["legend.fancybox"] = False
rcParams["legend.frameon"] = True
rcParams["legend.framealpha"] = 1.0
rcParams["legend.edgecolor"] = "black"
rcParams["font.size"] = 10

MARKERS = ["o", "s", "^", "v", "D", "P"]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def expand(paths):
    """Expand globs and return a flat sorted list of existing files."""
    out = []
    for p in paths:
        out.extend(glob.glob(p))
    return sorted(set(out))


def read_rows(paths):
    """Read one or more scaling CSVs into a list of dicts.

    Only the columns nx, ny, np, steps, wall_time_s, mlups are required;
    extra columns (e.g. nodes, ranks_per_node in the multi-node CSV) are
    ignored.
    """
    rows = []
    for path in expand(paths):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(
                    dict(
                        nx=int(r["nx"]),
                        ny=int(r["ny"]),
                        np=int(r["np"]),
                        steps=int(r["steps"]),
                        wall=float(r["wall_time_s"]),
                        mlups=float(r["mlups"]),
                    )
                )
    return rows


def group_by_size(rows):
    """Group rows by (nx, ny), each sorted ascending by np."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["nx"], r["ny"])].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r["np"])
    return dict(sorted(groups.items()))


def baseline(group):
    """Return the np==1 entry of a group (fallback: smallest np)."""
    for r in group:
        if r["np"] == 1:
            return r
    return group[0]


# --------------------------------------------------------------------------- #
# Strong scaling
# --------------------------------------------------------------------------- #
def plot_strong(rows, cores_per_socket, serial_fraction):
    groups = group_by_size(rows)
    all_p = np.array(sorted({r["np"] for r in rows}))

    fig, ax = plt.subplots(figsize=(5.0, 4.0), layout="constrained")

    print("\n=== STRONG SCALING (speedup S(P) = T(1)/T(P)) ===")
    for (size, group), marker in zip(groups.items(), MARKERS):
        b = baseline(group)
        P = np.array([r["np"] for r in group], dtype=float)
        T = np.array([r["wall"] for r in group], dtype=float)
        S = b["wall"] / T
        E = S / P

        print(f"\n  grid {size[0]}x{size[1]}:")
        print("     P   speedup   efficiency")
        for p, s, e in zip(P, S, E):
            print(f"  {int(p):4d}   {s:7.2f}   {100*e:6.1f}%")

        ax.plot(
            P, S, marker=marker, markersize=6, markeredgecolor="black",
            markeredgewidth=0.5, lw=1.4,
            label=f"{size[0]}×{size[1]}",
        )

    # Amdahl prediction. Profiling gave serial fraction ~ 0 -> near-linear.
    Pl = all_p.astype(float)
    amdahl = 1.0 / (serial_fraction + (1.0 - serial_fraction) / Pl)
    ax.plot(Pl, amdahl, "k--", lw=1.2,
            label=f"Amdahl (s={serial_fraction:.2f})")

    # Mark where ranks start to span the second socket.
    if cores_per_socket and cores_per_socket < all_p.max():
        ax.axvline(cores_per_socket, color="grey", ls=":", lw=1.0)
        ax.text(cores_per_socket * 1.02, ax.get_ylim()[0] * 1.5,
                "1 socket | 2 sockets", rotation=90,
                va="bottom", ha="left", fontsize=7, color="grey")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(all_p)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("number of MPI ranks $P$")
    ax.set_ylabel("speedup $S(P)$")
    ax.set_title("MPI strong scaling")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(title="grid $n_x\\times n_y$", fontsize=8)

    for ext in ("png", "pdf"):
        fn = f"strong_speedup.{ext}"
        fig.savefig(fn, dpi=150)
        print(f"  saved {fn}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Weak scaling
# --------------------------------------------------------------------------- #
def plot_weak(rows):
    rows = sorted(rows, key=lambda r: r["np"])
    P = np.array([r["np"] for r in rows], dtype=float)
    T = np.array([r["wall"] for r in rows], dtype=float)
    M = np.array([r["mlups"] for r in rows], dtype=float)
    E = T[0] / T  # weak efficiency: ideal => constant time => E = 1

    print("\n=== WEAK SCALING (efficiency E(P) = T(1)/T(P)) ===")
    print("     P     nx   wall(s)    MLUPS   efficiency")
    for r, e in zip(rows, E):
        print(f"  {r['np']:4d}  {r['nx']:6d}  {r['wall']:7.2f}  "
              f"{r['mlups']:8.1f}   {100*e:6.1f}%")

    fig, ax = plt.subplots(figsize=(5.0, 4.0), layout="constrained")
    ax.plot(P, E, marker="o", markersize=6, markeredgecolor="black",
            markeredgewidth=0.5, lw=1.4, label="measured")
    ax.axhline(1.0, color="black", ls="--", lw=1.2,
               label="ideal (Gustafson)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(P)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("number of MPI ranks $P$ (work $\\propto P$)")
    ax.set_ylabel("weak efficiency $T(1)/T(P)$")
    ax.set_title("MPI weak scaling")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize=8)

    for ext in ("png", "pdf"):
        fn = f"weak_efficiency.{ext}"
        fig.savefig(fn, dpi=150)
        print(f"  saved {fn}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Absolute throughput (bonus)
# --------------------------------------------------------------------------- #
def plot_mlups(strong_rows, weak_rows, cores_per_socket):
    fig, ax = plt.subplots(figsize=(5.0, 4.0), layout="constrained")

    groups = group_by_size(strong_rows)
    for (size, group), marker in zip(groups.items(), MARKERS):
        P = np.array([r["np"] for r in group], dtype=float)
        M = np.array([r["mlups"] for r in group], dtype=float)
        ax.plot(P, M, marker=marker, markersize=6, markeredgecolor="black",
                markeredgewidth=0.5, lw=1.4,
                label=f"strong {size[0]}×{size[1]}")

    if weak_rows:
        w = sorted(weak_rows, key=lambda r: r["np"])
        P = np.array([r["np"] for r in w], dtype=float)
        M = np.array([r["mlups"] for r in w], dtype=float)
        ax.plot(P, M, marker="*", markersize=9, color="black", lw=1.4,
                ls="--", label="weak (160k/rank)")

    all_p = np.array(sorted({r["np"] for r in strong_rows + weak_rows}))
    if cores_per_socket and cores_per_socket < all_p.max():
        ax.axvline(cores_per_socket, color="grey", ls=":", lw=1.0)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(all_p)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("number of MPI ranks $P$")
    ax.set_ylabel("throughput (MLUPS)")
    ax.set_title("MPI throughput")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize=7)

    for ext in ("png", "pdf"):
        fn = f"mlups_throughput.{ext}"
        fig.savefig(fn, dpi=150)
        print(f"  saved {fn}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strong", nargs="+", default=["strong_results_*.csv"],
                    help="strong-scaling CSV(s) or glob")
    ap.add_argument("--weak", nargs="+", default=["weak_results_*.csv"],
                    help="weak-scaling CSV(s) or glob")
    ap.add_argument("--cores-per-socket", type=int, default=36,
                    help="cores per socket (for the socket-boundary marker)")
    ap.add_argument("--serial-fraction", type=float, default=0.0,
                    help="Amdahl serial fraction from profiling (default 0)")
    args = ap.parse_args()

    strong_rows = read_rows(args.strong)
    weak_rows = read_rows(args.weak)

    if strong_rows:
        plot_strong(strong_rows, args.cores_per_socket, args.serial_fraction)
    else:
        print("No strong-scaling rows found.")

    if weak_rows:
        plot_weak(weak_rows)
    else:
        print("No weak-scaling rows found.")

    if strong_rows or weak_rows:
        plot_mlups(strong_rows, weak_rows, args.cores_per_socket)


if __name__ == "__main__":
    main()
