#!/bin/bash
#SBATCH --account=math-454
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=lbm-cuda-block
#SBATCH --output=block_%j.log
#SBATCH --time=00:20:00

# Block/grid study: fix the problem size + layout, vary the block shape.
# Principle to demonstrate: block_x should be a multiple of the 32-wide warp
# and lie along contiguous memory (x) for coalesced loads; very tall/thin or
# tiny blocks underutilize the SMs.

set -e
module purge
module load gcc cuda hdf5
make

OUT=block_results_${SLURM_JOB_ID}.csv
echo "nx,ny,block_x,block_y,layout,steps,wall_time_s,mlups" > "$OUT"

NX=4096; NY=2048; STEPS=2000; LAYOUT=soa

# block_x x block_y (threads/block in parentheses, all <= 1024)
CONFIGS=(32x4 32x8 32x16 64x4 64x8 64x16 128x4 128x8 256x4 16x16 8x32 512x2)

for cfg in "${CONFIGS[@]}"; do
  BX=${cfg%x*}; BY=${cfg#*x}
  echo "--- block ${BX}x${BY} ---"
  srun ./lbm_cuda nx=$NX ny=$NY steps=$STEPS layout=$LAYOUT \
       block_x=$BX block_y=$BY every=0 probe=/dev/null \
    | tee /dev/stderr | grep '^CSV:' | sed 's/^CSV: //' >> "$OUT"
done

echo "Wrote $OUT"; cat "$OUT"
