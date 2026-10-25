#!/usr/bin/env bash
# Wrapper around FastAFD's submit_megamoe_m2n_b200.sbatch for the NVIDIA
# Computelab B200 cluster.
#
# The upstream sbatch hardcodes `--account=beta-users`, which this user does not
# hold, and `--qos=interactive-isolated-short`, which caps us at one concurrent
# job. Both are overridden on the sbatch command line so the repo tree stays
# clean (the sbatch records source_dirty / source_tree_sha256 in every result
# JSON, and we want that provenance to say the tree is pristine).
set -euo pipefail

export HOST_ROOT=/home/scratch.yuny_wwfo_1/megamoe
export SOURCE_ROOT=${HOST_ROOT}/FastAFD
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-/home/scratch.yuny_wwfo_1/containers/sglang-latest.sqsh}

ACCOUNT=${ACCOUNT:-wwfo-china}
QOS=${QOS:-batch-short}
# All three 8-GPU B200 SXM partitions; Slurm takes whichever frees a whole node
# first. They are routinely 100% allocated, and every job here is --exclusive.
PARTITION=${PARTITION:-b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb,b200@501-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb}
TIME_LIMIT=${TIME_LIMIT:-03:55:00}
JOB_NAME=${JOB_NAME:-megamoe}

mkdir -p "${HOST_ROOT}/logs" "${HOST_ROOT}/results"

# The upstream sbatch's srun sets no --cpu-bind, so Slurm's default binding is
# used. On nodes whose allocation is NUMA-split -- umb-b200-262 hands out two
# 48-CPU blocks -- that default lands outside the allocation and srun dies in
# seconds with "Unable to satisfy cpu bind request" (exit 64). Nodes like
# umb-b200-240 have a contiguous mask and happen to work, so this failure is
# node-dependent and looks random. Exported rather than patched into the sbatch
# so the measured tree stays pristine: every result JSON records source_dirty,
# and that provenance should keep saying false.
export SLURM_CPU_BIND=none

# Optional afterok gate. The B200 queue is ~87 deep and every job waits for a
# whole free node, so the sweeps are pre-queued behind the smoke job rather
# than submitted serially. afterok only catches hard crashes -- the 1-ULP
# numerical contract is recorded in the JSON, not raised -- so the smoke JSON
# is still inspected by hand before the sweeps are allowed to stand.
dep_arg=()
if [[ -n ${DEPENDENCY:-} ]]; then
  dep_arg=(--dependency="${DEPENDENCY}")
fi

sbatch \
  "${dep_arg[@]}" \
  --account="${ACCOUNT}" \
  --qos="${QOS}" \
  --partition="${PARTITION}" \
  --time="${TIME_LIMIT}" \
  --job-name="${JOB_NAME}" \
  --output="${HOST_ROOT}/logs/${JOB_NAME}_%j.out" \
  --error="${HOST_ROOT}/logs/${JOB_NAME}_%j.err" \
  "${SOURCE_ROOT}/scripts/experiments/afd/submit_megamoe_m2n_b200.sbatch"
