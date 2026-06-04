# PHPC Final Project — LBM 2D Flow Past a Cylinder: Complete Handoff

**Course:** MATH-454 Parallel and High Performance Computing (EPFL), Dr. Pablo Antolín
**Task:** Parallelize a provided serial 2D Lattice Boltzmann (D2Q9 BGK) solver in two ways —
**(1) distributed-memory MPI**, **(2) single-GPU CUDA** — and produce a performance study.
**Deadline:** June 7, 23:59 CEST. **Validation target:** Strouhal number St ≈ 0.16 at Re = 100.

This document is a self-contained record of everything done, so any agent/person can recreate
or explain the work. It is a development doc, **not** part of the Moodle submission.

---

## 1. The physics & numerics (what the solver computes)

- **D2Q9 lattice Boltzmann, BGK single-relaxation-time collision.** Nine discrete velocities
  `c_i`; distributions `f_i(x,t)`. Indexing convention (from provided `lbm.cc`):
  `0`=rest, `1`=E, `2`=N, `3`=W, `4`=S, `5`=NE, `6`=NW, `7`=SW, `8`=SE.
  Opposites `opp = {0,3,4,1,2,7,8,5,6}`. Weights `w0=4/9, w1..4=1/9, w5..8=1/36`.
- **Macroscopic:** `rho = Σ f_i`, `rho*u = Σ c_i f_i`.
- **Equilibrium:** `f_i^eq = w_i ρ [1 + 3(c_i·u) + 4.5(c_i·u)² − 1.5|u|²]`.
- **Per step:** collide (relax toward `f^eq` by `1/τ`) → bounce-back in solids → stream
  (`f_i(x+c_i, t+1) = f_i^*(x,t)`) → inlet (left col = equilibrium at `u_in`) → outlet
  (zero-gradient: copy 2nd-to-last column to last).
- **Physics params:** `ν = c_s²(τ − 1/2)`, `c_s² = 1/3`, `Re = u_in·D/ν`. Default geometry
  `nx=800, ny=400, Re=100, u_in=0.05, cyl_r=10` (D=20) at `(nx/4, ny/2)`. No-slip top/bottom
  walls; one cylinder obstacle. At Re=100 → Kármán vortex street, **St ≈ 0.16**.
- **Validation tool:** `viz/strouhal.py probe.csv --u-in 0.05 --diameter 20` (FFT of probe `uy`).
  `viz/viz.py out/lbm.xdmf` renders the vorticity GIF.

Provided serial code: `lbm.{hh,cc}` (SoA storage `f_[i*nx*ny + y*nx + x]`, sweeps
`collide/bounce_back/stream/apply_inlet/apply_outlet`), `main.cc` (key=value CLI, prints MLUPS),
`output.{hh,cc}` (HDF5 + XDMF), `Makefile`, `viz/`.

---

## 2. Compute environment (SCITAS)

| | **JED** (MPI / CPU) | **izar** (CUDA / GPU) |
|---|---|---|
| Hardware | 2× Intel Xeon Platinum 8360Y (Ice Lake-SP), **72 cores/node** (2 sockets × 36, no HT) | NVIDIA **Tesla V100-PCIE-32GB** (Volta, **sm_70**), 80 SMs, 32 GB HBM2 |
| Mem bandwidth | ~140–150 GB/s achievable **per socket** (8-ch DDR4-3200) | **~900 GB/s** HBM2 |
| Toolchain | `module load gcc openmpi hdf5` (gcc 13.2, openmpi 5.0.3, hdf5 1.14.3) | `module load gcc cuda hdf5` (CUDA 11.8) |
| SLURM | `--account=math-454 --partition=academic` | `--account=math-454 --partition=gpu --gres=gpu:1` |

**Rule:** all builds/runs happen on the clusters (Mac is editor-only). CPU work on JED, GPU work on izar.

---

## 3. Serial baseline & profiling (Phase 0, on JED, 1 core)

**MLUPS** (mega lattice updates/s), 5000 steps, I/O off:

| Grid | Cells | MLUPS |
|---|---|---|
| 800×400 | 0.32 M | 21.75 |
| 1600×800 | 1.28 M | 21.35 |
| 3200×1600 | 5.12 M | 21.28 |

Flat across sizes ⇒ **memory-bandwidth bound** (no cache reuse to exploit).

**`perf` profile** (3000 steps, 800×400), self %:

| Routine | Self % | Parallel? |
|---|---|---|
| `collide()` | **71.4** | yes (per-cell) |
| `stream()` | **27.0** | yes (per-cell + 1-neighbour) |
| `bounce_back()` | 1.2 | yes |
| `apply_inlet()` | 0.24 | yes (rank 0 only) |
| `apply_outlet()` | <0.01 | yes (rank P−1 only) |

⇒ **Amdahl serial fraction ≈ 0**. The real ceiling is memory bandwidth + MPI halo cost, not Amdahl.
I/O overhead measured ~7% (HDF5 every 500 steps) → **all scaling runs disable I/O** (`every=0`, `probe=/dev/null`).

**Serial validation:** St = **0.1600** (default geometry, 60k steps).

---

## 4. MPI version (`mpi/`)

### 4.1 Design — 1D slab decomposition
- Domain split along **x** across P ranks: `nx_local = nx/P` (+1 for the first `nx%P` ranks).
  `x_start = rank*base + min(rank, rem)`.
- Local storage per rank: **9 slabs of `(nx_local+2)×ny` doubles** (one **ghost column** each side).
  Local x: `0`=left ghost, `1..nx_local`=interior, `nx_local+1`=right ghost. SoA: `f[i*N + y*nx_p + x_loc]`.
- **Per step:** `collide → bounce_back → exchange_halos → stream → apply_inlet → apply_outlet`.
- **Halo exchange:** pack the 9×ny edge-column values into a contiguous buffer; **2× `MPI_Sendrecv`**
  (right neighbour, left neighbour); `MPI_PROC_NULL` at the physical ends (no-op). No derived datatypes.
- **Boundary ownership:** rank 0 → inlet (global x=0); rank P−1 → outlet (global x=nx−1);
  every rank → top/bottom walls + cylinder (mapped via global x = `x_start + x_loc − 1`).
- **Probe:** the single rank owning `(probe_x, probe_y)` writes the CSV.
- **Timing:** `MPI_Barrier` → `MPI_Wtime` → loop → `MPI_Barrier` → `MPI_Reduce(MAX)`. MLUPS over **global** `nx·ny`.
- **Output:** per-row `MPI_Gatherv` to rank 0, which recomputes vorticity and writes the same HDF5/XDMF
  as serial (validation runs only).

### 4.2 Files
`mpi/lbm.{hh,cc}` (class `LBM_MPI`), `mpi/main.cc`, `mpi/output.{hh,cc}`, `mpi/Makefile` (uses `mpicxx`),
`mpi/README.md`; sweep scripts `strong_scaling.sbatch`, `weak_scaling.sbatch`,
`weak_scaling_multinode.sbatch`, `scaling_extend_128.sbatch`; `plot_scaling.py`.

### 4.3 KEY BUG FIX (physics)
The serial code collides solid nodes; the MPI port initially did too and **the flow decayed to zero
by ~step 2000** (probe `ux,uy → 0`). Fix: **`if (solid_[k]) continue;` in `collide()`** — fullway
bounce-back must NOT collide solid nodes (relaxing toward an equilibrium from a meaningless solid-cell
state injects spurious momentum). After the fix: **St = 0.1600**. (Other build fixes: Makefile
`CXX := mpicxx` not `?=`, because the gcc module sets `CXX=g++`; `#include <cstdint>` in `output.hh`.)

### 4.4 Validation
**St = 0.1600 at 4 ranks** — a strong test because the cylinder at x=200 straddles the rank-0/rank-1
boundary and the wake crosses ranks 1→2→3, exercising halo exchange with an obstacle on a seam.

### 4.5 Strong scaling (3 sizes, P=1..64, single node)
Speedup `S=T(1)/T(P)`, efficiency `E=S/P`:

| P | 800×400 E | 1600×800 E | 3200×1600 E |
|---|---|---|---|
| 1 | 100% | 100% | 100% |
| 2 | **103.5%** | 100.3% | 99.1% |
| 4 | **105.7%** | 94.4% | 95.0% |
| 8 | **104.0%** | 86.5% | 86.6% |
| 16 | **101.4%** | 74.8% | 71.8% |
| 32 | **102.3%** | 52.7% | 51.2% |
| 64 | 79.8% | 60.1% | 42.4% |

Peak **1235 MLUPS** (800×400, P=64). **Findings:**
- **Super-linear** on 800×400 (E>100%, P=2..32): fixed working set becomes **cache-resident** as each
  rank's slab shrinks — DRAM traffic collapses. Not an Amdahl violation.
- **Size-dependent**: bigger grids never fit cache → bandwidth-bound → graceful sub-linear.
- **Socket crossing**: 1600×800 efficiency **rises** 52.7%→60.1% at P=32→64 (2nd memory controller).

### 4.6 Weak scaling (160k cells/rank: ny=400, nx=400·P, steps=2000)
Efficiency `E=T(1)/T(P)` (ideal = constant time):

| P | nx | wall (s) | MLUPS | E |
|---|---|---|---|---|
| 1 | 400 | 12.13 | 26.4 | 100% |
| 2 | 800 | 12.34 | 51.8 | 98.3% |
| 4 | 1600 | 13.78 | 92.9 | 88.0% |
| 8 | 3200 | 15.45 | 165.7 | 78.5% |
| 16 | 6400 | 18.20 | 281.2 | 66.6% |
| 32 | 12800 | 23.35 | 438.5 | 52.0% |
| 64 | 25600 | 25.81 | 793.5 | 47.0% |

**Findings:** falls to 47% — **memory-bandwidth saturation** of one socket's fixed bus. Per-doubling
MLUPS multiplier **declines** within socket 0 (1.97→1.79→1.78→1.70→**1.56**) then **recovers to 1.81**
at 32→64 when ranks spill onto socket 1 (a second memory controller). Bandwidth check: P=32 →
438.5 MLUPS × 288 B = **126 GB/s** ≈ one socket's ceiling (saturated); P=64 → **229 GB/s** (two
sockets). Gustafson predicts throughput ∝ P; measured 30× at P=64 — the gap **is** the fixed bandwidth.
A node's memory bandwidth doesn't grow with P, so single-node weak scaling can't follow Gustafson.

`scaling_extend_128.sbatch` (2 nodes → P=128) and `weak_scaling_multinode.sbatch` exist to extend
across nodes; optional (queued behind QOS node limits).

---

## 5. CUDA version (`cuda/`)

### 5.1 Design — single GPU
- **One thread per cell**, 2D bounds-checked grid (`dim3 block(bx,by)`, ceil-div grid). **Double precision.**
- **Two layouts** via a compile-time `Layout` template parameter (zero runtime cost):
  **SoA** `f[i*N + cell]` (coalesced) vs **AoS** `f[cell*9 + i]` (stride-9). `fidx<L>()` folds to one address expr.
- **Kernels:** `init`, `collide` (with the solid-skip fix), `bounce_back`, `stream` (ping-pong + pointer
  swap), `inlet`, `outlet`, `probe`. D2Q9 constants in `__constant__` memory.
- **Timing:** CUDA events around the step loop with `cudaDeviceSynchronize()` before start — excludes the
  one-time H2D/D2H. `MLUPS = nx·ny·steps / time`.
- **Output:** host mirror `h_f_` (**lazily allocated** in `sync_to_host()` so timing runs on huge grids
  don't allocate multi-GB host RAM); reuses the serial HDF5/XDMF format. Probe = a 2-double D2H per step
  (disabled in timing runs via `probe=/dev/null`).

### 5.2 Files
`cuda/lbm.{hh,cu}` (class `LBM_CUDA` + templated kernels), `cuda/main.cu`, `cuda/output.{hh,cc}`,
`cuda/Makefile` (`nvcc -O3 -arch=sm_70 -std=c++14 -lhdf5`), `cuda/README.md`; sweeps
`sweep_block.sbatch`, `sweep_size.sbatch`, `sweep_layout.sbatch`, `sweep_fused.sbatch`; `plot_cuda.py`.
CLI adds `block_x`, `block_y`, `layout` (soa/aos), `fused` (0/1).

### 5.3 Validation
**St = 0.1600** for both the separate-kernel and fused versions.

### 5.4 Performance results (V100)
**Roofline:** 900 GB/s ÷ 288 B/cell = **3125 MLUPS** ceiling (separate kernels).

- **Block/grid** (4096×2048, SoA): ~2500–2600 MLUPS for any block with **block_x ≥ 32** (one warp);
  thin-in-x outliers lose coalescing (8×32 → 1910, 16×16 → 2370). Performance is robust to block
  shape above the warp width.
- **Problem size** (SoA, 32×8): 1930 (128k cells) → ~2550 (8M) → 2510 (33M). Small grids underfill the
  80 SMs (low occupancy); saturates at **~82% of roofline**. Best ≈ **2610 MLUPS** (≈ **83%** = 751 GB/s).
- **Memory layout** (32×8): **SoA ~2500 vs AoS ~640 MLUPS — a ~4× gap.** SoA ≈ 80% of bandwidth,
  AoS ≈ 20% (stride-9 access destroys coalescing). The clearest result in the study.

### 5.5 Going-further: fused collide-stream kernel
- **Idea:** the separate kernels make `f` round-trip to DRAM between them (collide writes, stream reads) →
  288 B/cell. A **fused kernel** does both in one pass (**144 B/cell**), doubling the roofline to **6250 MLUPS**.
- **Scheme:** **pull** + **post-collision storage** + **bounce-back-on-link**. Each fluid cell gathers the 9
  arriving distributions from neighbours (reflecting `opp[i]` at solid/boundary links), computes moments
  (valid — BGK conserves ρ and momentum), relaxes, writes its own 9 values. 9 reads + 9 writes, one pass.
- **Results** (32×8, SoA): separate vs fused MLUPS —
  1024×512: 2279 → **3548** (1.56×); 2048×1024: 2481 → **3951** (1.59×);
  4096×2048: 2558 → **3965** (1.55×); 8192×4096: 2513 → **3937** (1.57×).
- **Outcome:** **~1.55× speedup** (≈2550 → ≈3950 MLUPS). Predicted 2×, achieved 1.55× — the fused kernel
  reaches only **63%** of its 6250 ceiling (vs separate's 82% of 3125) because the scattered pull-reads
  coalesce less cleanly and higher register pressure lowers occupancy. **Validated St = 0.1600.**
- CLI: `fused=1`. (Note: fused uses *halfway* bounce-back-on-link vs the separate path's *fullway* swap —
  both valid no-slip; both give St=0.16.)

---

## 6. Headline comparison

| Version | MLUPS | × serial | % of its roofline |
|---|---|---|---|
| Serial (1 Xeon core) | 21.5 | 1× | — |
| MPI (64 ranks, 1 node) | 1235 | 57× | — |
| CUDA separate (1 V100) | 2610 | 121× | 83% (3125) |
| **CUDA fused (1 V100)** | **~3950** | **184×** | 63% (6250) |

**Unifying theme:** every version is **memory-bandwidth bound**. The whole performance story is about
moving fewer bytes or getting more bandwidth — cache residency (super-linear strong scaling), socket
crossing (weak scaling), HBM2 vs DDR4 (GPU vs CPU), and fusing (halving GPU traffic).

---

## 7. Reproduction quickstart

```bash
# --- Serial (JED) ---
module load gcc hdf5 && make
srun --account=math-454 --partition=academic -n1 ./lbm steps=60000 every=500
python3 viz/strouhal.py probe.csv --u-in 0.05 --diameter 20   # -> St 0.1600

# --- MPI (JED) ---
cd mpi && module load gcc openmpi hdf5 && make
sbatch strong_scaling.sbatch     # strong_results_<jid>.csv
sbatch weak_scaling.sbatch       # weak_results_<jid>.csv
python3 plot_scaling.py          # strong_speedup.* weak_efficiency.* mlups_throughput.*

# --- CUDA (izar) ---
cd cuda && module load gcc cuda hdf5 && make
sbatch sweep_block.sbatch; sbatch sweep_size.sbatch; sbatch sweep_layout.sbatch; sbatch sweep_fused.sbatch
python3 plot_cuda.py size_results_*.csv     # auto-detects sweep type, draws roofline
# validate: ./lbm_cuda steps=60000 every=0 probe=p.csv fused=1 ; strouhal.py p.csv ... -> 0.1600
```

Scaling/timing runs use `every=0 probe=/dev/null`. Block sweeps fix SoA 32×8. V100 roofline numbers:
900 GB/s, 288 B/cell (separate) / 144 B/cell (fused).

---

## 8. What remains (not yet done)

1. **Report** (PDF, ≤1500 words body, **≤5 figures/tables**, + **AI usage statement**). Planned figures:
   (1) MPI strong scaling, (2) MPI weak scaling, (3) CUDA size+roofline, (4) CUDA layout or fused,
   (5) validation vorticity snapshot.
2. **Slides** (PDF, exactly 4): strong scaling, weak scaling, CUDA performance, free choice
   (candidate: combined CPU+GPU roofline, or the fused-kernel result).
3. **Tarball** for Moodle: two clearly separated folders (`mpi/`, `cuda/`), each with its README.
4. (Optional) recover the 12288×6144 fused/separate data point (rebuild picks up the lazy-alloc fix).

---

## 9. Gotchas / lessons (for whoever continues)

- **Don't collide solid nodes** in fullway bounce-back — it silently kills the flow.
- MPI Makefile must **force** `CXX := mpicxx` (gcc module exports `CXX=g++`).
- Time MPI with `MPI_Wtime` + `MPI_Reduce(MAX)` (slowest rank = wall time); compute MLUPS on **global** size.
- Time CUDA with **cudaEvents** + a sync before the start; exclude H2D/D2H; `probe=/dev/null` for timing.
- Default SLURM `--mem` can be small — lazily allocate big host buffers, or request memory.
- Super-linear speedup and >100% / socket-crossing recoveries are **real cache/bandwidth effects** — flag
  and explain them (the rubric explicitly rewards investigating anomalies).
- SoA is essential on GPU (coalescing); AoS is ~4× slower and is the layout-study control.
