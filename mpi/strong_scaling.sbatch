#!/bin/bash
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=lbm-strong
#SBATCH --output=strong_%j.log
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

# Strong scaling: fix the problem size, vary the number of ranks.
# Speedup S(P) = T(1) / T(P) is computed afterwards from the CSV.
#
# Three representative sizes. Per-size step counts keep the P=1 baseline
# to a few minutes (MLUPS is normalised by cells*steps, so different step
# counts across sizes do not bias the comparison; within a size the step
# count is fixed, which is what the speedup ratio requires).

set -e

module purge
module load gcc openmpi hdf5

echo "Host  : $(hostname)"
echo "Date  : $(date)"
echo "Cores : $(nproc)"
echo

make clean && make

RESULTS=strong_results_${SLURM_JOB_ID}.csv
echo "nx,ny,np,steps,wall_time_s,mlups" > "$RESULTS"

PROCS=(1 2 4 8 16 32 64)

# size label : "nx ny steps"
declare -a CONFIGS=(
  "800 400 4000"
  "1600 800 2000"
  "3200 1600 1000"
)

for cfg in "${CONFIGS[@]}"; do
  read -r NX NY STEPS <<< "$cfg"
  echo "=================================================="
  echo "Strong scaling for ${NX} x ${NY}, steps=${STEPS}"
  echo "=================================================="
  for NP in "${PROCS[@]}"; do
    echo "--- np=${NP} ---"
    srun -n "$NP" --cpu-bind=cores \
      ./lbm_mpi nx=$NX ny=$NY steps=$STEPS every=0 probe=/dev/null \
      | tee /dev/stderr \
      | grep '^CSV:' | sed 's/^CSV: //' >> "$RESULTS"
    echo
  done
done

echo "Wrote $RESULTS"
cat "$RESULTS"
