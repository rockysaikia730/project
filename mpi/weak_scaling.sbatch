#!/bin/bash
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=lbm-weak
#SBATCH --output=weak_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

# Weak scaling: fix the work PER rank, grow the total problem with P.
#
# 1D slab decomposition is along x, so we keep ny fixed and set
# nx = NX_PER_RANK * P. Each rank then always owns NX_PER_RANK * ny cells,
# and the halo it exchanges is always ny cells per face. Ideal (Gustafson)
# behaviour is constant wall time => MLUPS growing linearly with P.

set -e

module purge
module load gcc openmpi hdf5

echo "Host  : $(hostname)"
echo "Date  : $(date)"
echo "Cores : $(nproc)"
echo

make clean && make

RESULTS=weak_results_${SLURM_JOB_ID}.csv
echo "nx,ny,np,steps,wall_time_s,mlups" > "$RESULTS"

PROCS=(1 2 4 8 16 32 64)

NY=400
NX_PER_RANK=400      # each rank owns 400 x 400 = 160k cells
STEPS=2000

for NP in "${PROCS[@]}"; do
  NX=$(( NX_PER_RANK * NP ))
  echo "=================================================="
  echo "Weak scaling: np=${NP}, nx=${NX} (${NX_PER_RANK}/rank), ny=${NY}, steps=${STEPS}"
  echo "=================================================="
  srun -n "$NP" --cpu-bind=cores \
    ./lbm_mpi nx=$NX ny=$NY steps=$STEPS every=0 probe=/dev/null \
    | tee /dev/stderr \
    | grep '^CSV:' | sed 's/^CSV: //' >> "$RESULTS"
  echo
done

echo "Wrote $RESULTS"
cat "$RESULTS"
