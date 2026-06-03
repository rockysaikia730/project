#!/bin/bash
#SBATCH --account=math-454
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=lbm-cuda-size
#SBATCH --output=size_%j.log
#SBATCH --time=00:30:00

# Problem-size study: fix block + layout, grow the grid. Small grids do not
# launch enough blocks to fill the V100's 80 SMs (low occupancy -> low MLUPS);
# large grids saturate memory bandwidth and approach the roofline. Largest grid
# (16384x8192 = 134M cells) needs ~19 GB on device (f + ftmp), safe on 32 GB.

set -e
module purge
module load gcc cuda hdf5
make

OUT=size_results_${SLURM_JOB_ID}.csv
echo "nx,ny,block_x,block_y,layout,steps,wall_time_s,mlups" > "$OUT"

BX=32; BY=8; LAYOUT=soa; STEPS=2000

CONFIGS=(
  "512 256"
  "1024 512"
  "2048 1024"
  "4096 2048"
  "8192 4096"
  "12288 6144"
  "16384 8192"
)

for cfg in "${CONFIGS[@]}"; do
  read -r NX NY <<< "$cfg"
  echo "--- ${NX}x${NY} ---"
  srun ./lbm_cuda nx=$NX ny=$NY steps=$STEPS layout=$LAYOUT \
       block_x=$BX block_y=$BY every=0 probe=/dev/null \
    | tee /dev/stderr | grep '^CSV:' | sed 's/^CSV: //' >> "$OUT"
done

echo "Wrote $OUT"; cat "$OUT"
