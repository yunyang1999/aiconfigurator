#!/usr/bin/env bash
# Stage and run FastAFD on ComputeLab GB200 nodes.
#
# GB200 is aarch64 (Grace) with CUDA 13, which is the only platform FastAFD
# declares validated. This box (x86_64, H20 = sm90) cannot run it at all: the
# AFD path is `sm100_mega_moe_m2n.hpp` / `sm100_fp8_fp4_mega_moe.hpp`, Blackwell
# only.
#
# Prerequisite: key-based login to the frontend must already work, i.e.
#   ssh -o BatchMode=yes yuny@computelab-sc-01.nvidia.com true
# must succeed. Everything below runs FROM the frontend.
#
# Usage:  bash computelab_gb200_fastafd.sh discover
#         bash computelab_gb200_fastafd.sh alloc   <n_nodes> <partition>
#         bash computelab_gb200_fastafd.sh setup                 # on an allocated node
#         bash computelab_gb200_fastafd.sh align                 # 2-node smoke test
#         bash computelab_gb200_fastafd.sh bench235b <n_nodes>   # published preset
set -euo pipefail

FRONTEND="${FRONTEND:-computelab-sc-01.nvidia.com}"
SCRATCH="${SCRATCH:-/home/scratch.${USER}_sw}"
WORK="${WORK:-$SCRATCH/fastafd}"
ENVNAME="${ENVNAME:-minisgl-cuda130}"

case "${1:-}" in

discover)
  # Which GB200 nodes exist, and which are free. Partition names follow
  #   {gpu_count}gpu-{product}-{trainstop}_{server}-{trainstop}-{cores}-{mem}gb
  echo "--- GB200-ish nodes ---"
  cdb find --fields "hostname,gpu.product_name,system.gpu_count,cpu.arch,slurm.partitions,state" \
    | grep -iE "gb200|b200|blackwell" || echo "(none matched; widen the grep)"
  echo
  echo "--- free ones ---"
  cdb find --state "idle" --fields "hostname,gpu.product_name,system.gpu_count,slurm.partitions" \
    | grep -iE "gb200|b200" || true
  echo
  echo "--- high-speed networks (needed for multi-node) ---"
  cdb find --fields "hostname,system.ib_network_speed_gbps,partition,networks,state" --networks "ib*" \
    | grep -iE "gb200|b200" || true
  ;;

alloc)
  N="${2:?n_nodes}"; PART="${3:?partition}"
  # -N for node count, ib constraint so the nodes share a fabric, long walltime
  # because the build alone takes a while.
  echo "sbatch -N $N -p $PART --constraint=ib0 --time=8:00:00 ..."
  echo "or interactive:  crun -N $N -q \"gpu.product_name=GB200\" --network ib0 --time 8:00:00 -i"
  ;;

setup)
  mkdir -p "$WORK"; cd "$WORK"
  [ -d FastAFD ] || git clone https://github.com/hao-ai-lab/FastAFD.git
  cd FastAFD
  conda env create -f environment.cuda130.yml -n "$ENVNAME" 2>/dev/null || echo "env exists"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENVNAME"
  pip install -e .
  # aarch64 wheel -- matches GB200, which is why this only works here
  python -m pip install \
    "https://github.com/vllm-project/vllm/releases/download/v0.19.0/vllm-0.19.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu130
  python -m pip install --no-cache-dir --force-reinstall --no-deps \
    "nvidia-nccl-cu13==2.30.4" "setuptools==80.10.2" "fsspec==2026.2.0"
  python -c "import torch;print('torch',torch.__version__,'cap',torch.cuda.get_device_capability())"
  ;;

align)
  # Smallest end-to-end check: Qwen3-30B-A3B, attention TP4 + MLP TP4 = 8 GPUs
  # = 2 GB200 nodes. Needs a running Ray cluster (RAY_ADDRESS=auto).
  cd "$WORK/FastAFD"
  ./scripts/validate/qwen3_30b_a3b_fastafd_alignment.sh
  ;;

bench235b)
  N="${2:-4}"
  cd "$WORK/FastAFD"
  # 3 attn nodes + 1 MLP node at N=4. MODEL_PATH must be a shared snapshot on
  # scratch; per-rank HF cache resolution does not work multi-node.
  MODEL_PATH="${MODEL_PATH:?point at a local Qwen3-235B-A22B-FP8 snapshot}" \
  AFD_TOTAL_NODES="$N" RUN_VLLM_ALIGNMENT=0 NSYS=0 \
    bash scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh
  # Compare the reported decode-step latency against AIC:
  #   scripts/afd_fastafd/absolute_check.py  (FastAFD blog: 32.826 ms at 4-node 8K)
  ;;

*)
  sed -n '2,30p' "$0"; exit 1;;
esac
