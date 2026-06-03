# LBM 2D — CUDA version (single GPU, V100)

Single-GPU CUDA port of the serial solver. One thread per lattice cell, 2D
bounds-checked grid. Double precision. The nine distribution slabs are stored
either **SoA** (`f[i*N + cell]`, coalesced) or **AoS** (`f[cell*9 + i]`,
strided) — chosen at runtime, with the kernels templated on the layout so the
choice costs nothing in the inner loop.

## Build (on izar)

```bash
module load gcc cuda hdf5
make
```

`-arch=sm_70` targets the V100. `$HDF5_ROOT` (set by the hdf5 module) is picked
up automatically for the snapshot writer.

## Run

```bash
# Quick MLUPS check (no I/O), 32x8 blocks, SoA layout:
srun --account=math-454 --partition=gpu --gres=gpu:1 --time=00:05:00 \
     ./lbm_cuda nx=2000 ny=1000 steps=2000 every=0 probe=/dev/null

# Validation run: 60k steps, snapshots + probe -> St ~ 0.16
srun --account=math-454 --partition=gpu --gres=gpu:1 --time=00:15:00 \
     ./lbm_cuda steps=60000 every=500 probe=probe.csv
python3 ../viz/strouhal.py probe.csv --u-in 0.05 --diameter 20
python3 ../viz/viz.py out/lbm.xdmf            # vorticity.gif
```

## CLI keys

Same as serial, plus:

| key | meaning | default |
|---|---|---|
| `block_x`, `block_y` | CUDA block dimensions (threads/block = block_x*block_y) | 32, 8 |
| `layout` | `soa` or `aos` | `soa` |
| `fused` | `1` = fused collide-stream (144 B/cell), `0` = separate kernels (288 B/cell) | `0` |
| `every` | HDF5 snapshot cadence; 0 disables I/O | 0 |
| `probe` | probe CSV path; `/dev/null` disables the per-step D2H | probe.csv |

## Output

```
Wall time : <s>          (cudaEvent-timed step loop, excludes H2D/D2H)
MLUPS     : <value>
CSV: nx,ny,block_x,block_y,layout,steps,wall_time_s,mlups
```

The `CSV:` line is parsed by the sweep scripts (block/grid, problem size,
layout). Timing uses CUDA events around the step loop with a
`cudaDeviceSynchronize()` before the start event, so it measures GPU compute,
not the one-time host<->device transfers.

## Notes

- `collide` skips solid cells (`if (solid) return;`); fullway bounce-back then
  reverses their populations. This matches the validated MPI version.
- Streaming uses a ping-pong second buffer (`d_ftmp`) and a pointer swap.
- The probe copies just two doubles per step (cheap) and is disabled in the
  timing runs via `probe=/dev/null`.
- Roofline reference: V100 HBM2 ~900 GB/s, ~288 B/cell/step -> ~3100 MLUPS bound.
