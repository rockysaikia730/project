#!/bin/bash
#SBATCH --account=math-454
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=lbm-cuda-fused
#SBATCH --output=fused_%j.log
#SBATCH --time=00:30:00

# Going-further: fused collide-stream vs the separate-kernel baseline.
# Fusing halves DRAM traffic (288 -> 144 B/cell), so the roofline ceiling
# doubles (3125 -> 6250 MLUPS). This sweep runs both kernels across sizes at
# the best block (32x8, SoA) and prepends a "kernel" label to each CSV row.

set -e
module purge
module load gcc cuda hdf5
make

OUT=fused_results_${SLURM_JOB_ID}.csv
echo "kernel,nx,ny,block_x,block_y,layout,steps,wall_time_s,mlups" > "$OUT"

BX=32; BY=8; STEPS=2000
CONFIGS=("1024 512" "2048 1024" "4096 2048" "8192 4096" "12288 6144")

for cfg in "${CONFIGS[@]}"; do
  read -r NX NY <<< "$cfg"
  for K in separate fused; do
    FUSED=$([ "$K" = fused ] && echo 1 || echo 0)
    echo "--- ${NX}x${NY} ${K} ---"
    line=$(srun ./lbm_cuda nx=$NX ny=$NY steps=$STEPS layout=soa \
                 block_x=$BX block_y=$BY fused=$FUSED every=0 probe=/dev/null \
           | tee /dev/stderr | grep '^CSV:' | sed 's/^CSV: //')
    echo "${K},${line}" >> "$OUT"
  done
done

echo "Wrote $OUT"; cat "$OUT"
