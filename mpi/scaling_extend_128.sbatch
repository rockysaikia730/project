#!/bin/bash
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=lbm-ext128
#SBATCH --output=ext128_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

# Extend BOTH scaling studies to P=128 across 2 nodes (64 ranks/node, leaving
# 8 cores/node free for the OS + MPI daemons). Output goes to separate CSVs
# that plot_scaling.py picks up via its glob, so the curves simply gain a
# P=128 point.
#
#   Strong: the three fixed sizes, now at P=128. The small grid (800x400)
#           should finally roll over (halo/interior ~ 33%), showing the
#           plateau the single-node sweep stopped just short of.
#   Weak  : 160k cells/rank held fixed -> nx = 400 * 128 = 51200.

set -e

module purge
module load gcc openmpi hdf5

echo "Host        : $(hostname)"
echo "Date        : $(date)"
echo "Cores/node  : $(nproc)"
echo "Nodes alloc : $SLURM_JOB_NUM_NODES"
echo

make clean && make

# ---- strong: append P=128 to each size --------------------------------------
STRONG=strong_results_mn_${SLURM_JOB_ID}.csv
echo "nx,ny,np,steps,wall_time_s,mlups" > "$STRONG"

declare -a CONFIGS=(
  "800 400 4000"
  "1600 800 2000"
  "3200 1600 1000"
)
for cfg in "${CONFIGS[@]}"; do
  read -r NX NY STEPS <<< "$cfg"
  echo "=== strong ${NX}x${NY}, P=128 ==="
  srun -N 2 --ntasks=128 --ntasks-per-node=64 --cpu-bind=cores \
    ./lbm_mpi nx=$NX ny=$NY steps=$STEPS every=0 probe=/dev/null \
    | tee /dev/stderr | grep '^CSV:' | sed 's/^CSV: //' >> "$STRONG"
  echo
done

# ---- weak: append P=128 -----------------------------------------------------
WEAK=weak_results_mn_${SLURM_JOB_ID}.csv
echo "nx,ny,np,steps,wall_time_s,mlups" > "$WEAK"

NX_PER_RANK=400
NY=400
STEPS=2000
NX=$(( NX_PER_RANK * 128 ))   # 51200
echo "=== weak P=128, nx=${NX} (${NX_PER_RANK}/rank) ==="
srun -N 2 --ntasks=128 --ntasks-per-node=64 --cpu-bind=cores \
  ./lbm_mpi nx=$NX ny=$NY steps=$STEPS every=0 probe=/dev/null \
  | tee /dev/stderr | grep '^CSV:' | sed 's/^CSV: //' >> "$WEAK"

echo
echo "Wrote $STRONG and $WEAK"
cat "$STRONG"; echo; cat "$WEAK"
