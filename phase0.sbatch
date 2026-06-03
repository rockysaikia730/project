#!/bin/bash
#SBATCH --account=math-454
#SBATCH --job-name=lbm-phase0
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=phase0_%j.log

# Phase 0: serial baseline + validation + profile.
# Single job, single CPU. All runs serial — no MPI yet.
#
# What this produces:
#   - probe.csv          : 60k-step validation run, fed to viz/strouhal.py
#   - out/lbm.xdmf + *.h5: snapshots from the validation run, for vorticity.gif
#   - phase0_*.log       : MLUPS at three problem sizes + perf hot-spot report

set -e
set -o pipefail

# Modules (same as the build step)
module purge
module load gcc
module load hdf5

echo "=========================================="
echo "Host    : $(hostname)"
echo "Date    : $(date)"
echo "CPU info: $(grep 'model name' /proc/cpuinfo | head -1 | sed 's/.*: //')"
echo "=========================================="

# ----------------------------------------------------------------------
# Step 0.3: Validation run.
# Default geometry (800x400, Re=100, cyl_r=10). Full 60k steps so Karman
# shedding develops and the FFT in strouhal.py finds a clean peak.
# ----------------------------------------------------------------------
echo
echo "### Step 0.3: Validation run (60k steps, default geometry) ###"
rm -rf out && mkdir out
./lbm steps=60000 every=500 probe=probe.csv out=out/lbm

# ----------------------------------------------------------------------
# Step 0.5: Baseline MLUPS at three problem sizes.
# HDF5 off (every=0); probe written to /dev/null to minimize I/O noise.
# 5000 steps is enough to amortize startup at all three sizes.
# ----------------------------------------------------------------------
echo
echo "### Step 0.5: Baseline MLUPS at three sizes (5000 steps, no I/O) ###"
for cfg in "nx=800 ny=400" "nx=1600 ny=800" "nx=3200 ny=1600"; do
  echo "--- $cfg ---"
  ./lbm $cfg steps=5000 every=0 probe=/dev/null 2>&1 \
      | grep -E "(grid|tau|Wall time|MLUPS)"
  echo
done

# ----------------------------------------------------------------------
# Step 0.4: Profile with perf (Linux sampling profiler).
# Falls back to a note if perf is restricted on JED compute nodes.
# ----------------------------------------------------------------------
echo
echo "### Step 0.4: perf profile (3000 steps, 800x400) ###"
if command -v perf >/dev/null 2>&1; then
  if perf record -F 199 -g -o phase0.perf -- \
       ./lbm steps=3000 every=0 probe=/dev/null 2>perf_record.err; then
    echo "--- perf report (top functions, no children) ---"
    perf report -i phase0.perf --stdio --no-children 2>/dev/null | head -40
    echo "--- perf report (with children, call-stack rolled up) ---"
    perf report -i phase0.perf --stdio 2>/dev/null | head -40
  else
    echo "perf record failed:"
    cat perf_record.err
    echo "(May need different paranoia level or a perf-enabled partition.)"
  fi
else
  echo "perf not found in PATH on this node."
fi

echo
echo "=== phase0 complete ==="
