#!/bin/bash
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --job-name=lbm-weak-mn
#SBATCH --output=weak_mn_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

# Multi-node weak scaling: fix RANKS_PER_NODE, scale the number of NODES.
#
# Purpose: isolate intra-node memory-bandwidth saturation from the parallel
# algorithm. The single-node weak study showed efficiency falling to ~52% at
# P=32 because all ranks share one node's fixed memory bandwidth. Here we keep
# 16 ranks per node and add nodes, so each new rank brings additional memory
# bandwidth. The decisive comparison:
#
#   32 ranks on 1 node  (from weak_scaling.sbatch)  vs
#   32 ranks on 2 nodes (16/node, this script)
#
# If the 2-node point is markedly more efficient, the single-node degradation
# was memory bandwidth, not communication or load imbalance.

set -e

module purge
module load gcc openmpi hdf5

echo "Host  : $(hostname)"
echo "Date  : $(date)"
echo "Cores/node : $(nproc)"
echo "Nodes allocated : $SLURM_JOB_NUM_NODES"
echo

make clean && make

RESULTS=weak_mn_results_${SLURM_JOB_ID}.csv
echo "nx,ny,np,nodes,ranks_per_node,steps,wall_time_s,mlups" > "$RESULTS"

RANKS_PER_NODE=16
NY=400
NX_PER_RANK=400
STEPS=2000

for N in 1 2 4; do
  NP=$(( RANKS_PER_NODE * N ))
  NX=$(( NX_PER_RANK * NP ))
  echo "=================================================="
  echo "nodes=${N}, ranks/node=${RANKS_PER_NODE}, total P=${NP}, nx=${NX}"
  echo "=================================================="
  # Force exactly RANKS_PER_NODE ranks on each of N nodes.
  line=$(srun -N "$N" --ntasks="$NP" --ntasks-per-node="$RANKS_PER_NODE" \
              --cpu-bind=cores \
         ./lbm_mpi nx=$NX ny=$NY steps=$STEPS every=0 probe=/dev/null \
         | tee /dev/stderr | grep '^CSV:' | sed 's/^CSV: //')
  # CSV from program is nx,ny,np,steps,wall,mlups -> splice in nodes + rpn.
  IFS=',' read -r cnx cny cnp csteps cwall cmlups <<< "$line"
  echo "${cnx},${cny},${cnp},${N},${RANKS_PER_NODE},${csteps},${cwall},${cmlups}" >> "$RESULTS"
  echo
done

echo "Wrote $RESULTS"
cat "$RESULTS"
