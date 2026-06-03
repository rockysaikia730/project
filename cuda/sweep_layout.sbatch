#!/bin/bash
#SBATCH --account=math-454
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=lbm-cuda-layout
#SBATCH --output=layout_%j.log
#SBATCH --time=00:30:00

# Memory-layout study: SoA vs AoS at a fixed block, across several sizes.
# With one thread per cell, SoA (f[i*N + cell]) gives coalesced loads -
# adjacent threads (adjacent x) touch adjacent addresses within each slab.
# AoS (f[cell*9 + i]) scatters those by a stride of 9, wasting bandwidth.
# Expect SoA to win by a wide, size-independent margin.

set -e
module purge
module load gcc cuda hdf5
make

OUT=layout_results_${SLURM_JOB_ID}.csv
echo "nx,ny,block_x,block_y,layout,steps,wall_time_s,mlups" > "$OUT"

BX=32; BY=8; STEPS=2000

CONFIGS=(
  "1024 512"
  "2048 1024"
  "4096 2048"
  "8192 4096"
)

for cfg in "${CONFIGS[@]}"; do
  read -r NX NY <<< "$cfg"
  for LAYOUT in soa aos; do
    echo "--- ${NX}x${NY} ${LAYOUT} ---"
    srun ./lbm_cuda nx=$NX ny=$NY steps=$STEPS layout=$LAYOUT \
         block_x=$BX block_y=$BY every=0 probe=/dev/null \
      | tee /dev/stderr | grep '^CSV:' | sed 's/^CSV: //' >> "$OUT"
  done
done

echo "Wrote $OUT"; cat "$OUT"
