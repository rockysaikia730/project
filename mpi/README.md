# LBM 2D — MPI version

Distributed-memory parallel version of the serial solver in the parent
directory. 1D slab decomposition along x; each rank owns a contiguous
range of columns plus a one-cell ghost layer on each side. Halos are
exchanged once per step with two `MPI_Sendrecv` calls (pack/unpack into
contiguous buffers, no derived datatypes).

## Build (on JED)

```bash
module load gcc openmpi hdf5
make
```

`HDF5_ROOT` is set automatically by the `hdf5` module. The Makefile picks
it up.

## Run

```bash
# Single-node, 8 ranks, default geometry, no I/O — for a quick MLUPS check.
srun --account=math-454 --partition=academic --time=00:05:00 \
     --nodes=1 --ntasks=8 --cpus-per-task=1 \
     ./lbm_mpi steps=2000 every=0 probe=/dev/null

# Full validation run: 60k steps with HDF5 snapshots + probe CSV.
srun --account=math-454 --partition=academic --time=00:30:00 \
     --nodes=1 --ntasks=8 --cpus-per-task=1 \
     ./lbm_mpi steps=60000 every=500 probe=probe.csv
```

Validate the Karman shedding with the same script as serial:

```bash
python3 ../viz/strouhal.py probe.csv --u-in 0.05 --diameter 20
# expect St ~ 0.16
```

## CLI keys

Same as the serial program. Extra notes for MPI:

| key | meaning |
|---|---|
| `nx`, `ny` | global grid size |
| `steps`, `every` | as in serial. `every=0` disables HDF5 I/O. |
| `probe` | output CSV path. Only the rank that owns `(probe_x, probe_y)` writes; pass `/dev/null` to disable. |

## Output

At the end of every run, rank 0 prints two lines used by the analysis scripts:

```
Wall time : <seconds>
MLUPS     : <millions of lattice updates per second>
CSV: nx,ny,np,steps,wall_time_s,mlups
```

The `CSV:` line is the one parsed by the strong/weak-scaling sweep scripts.

## Implementation notes

- `lbm.{cc,hh}` — `LBM_MPI` class. Same five sweeps as serial, with `exchange_halos()` inserted between `bounce_back()` and `stream()`. Local indexing has ghost columns at `x_loc = 0` and `x_loc = nx_local + 1`.
- `main.cc` — `MPI_Init`/`Finalize`, parse `key=value` args, run loop, timing via `MPI_Barrier` + `MPI_Wtime` + `MPI_Reduce(MAX)`.
- `output.cc` — per-row `MPI_Gatherv` of the local field slabs to rank 0; rank 0 computes vorticity from the gathered velocity and writes HDF5+XDMF in the same format as the serial writer.
- Boundary ownership: rank 0 applies the inlet (global `x=0`), rank `P-1` applies the outlet (global `x=nx-1`), every rank applies the no-slip top/bottom walls and the cylinder bounce-back locally.
