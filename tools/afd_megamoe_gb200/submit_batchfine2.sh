#!/usr/bin/env bash
# Round 2: refine where the capacity wall actually falls at LONG context.
#
# Round 1 added rungs 72..224 and made the Qwen3 isl=65536 comparison WORSE, not
# better: AFD moved 96 -> 112 and used them, colocated stayed at 64 and did not.
# The reason is that the two arms are parameterised on different quantities.
#
#   colocated  logical_tokens_per_rank IS the per-GPU concurrency, so its wall
#              sits at seq/GPU, which scales as 1/isl. Qwen3 at isl=65536 walls
#              at 27, at 131072 at ~13.5, at 32768 at ~54.
#   AFD        sequences_per_ag_rank is per A rank and the A pool is wide, so its
#              wall sits much higher -- 118 for 68A4F/tp_a4 at isl=65536.
#
# Round 1 covered the AFD side and the short-context colocated side. What it
# missed is the small end, which is exactly where the long-context colocated wall
# lives. These rungs bracket 13.5, 27 and 54 from both sides.
#
# All even, so the microbatch=2 suite can use them too.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GRID="10 12 14 20 24 28 36 40 44 52 56 60"

for ep in 1 2 4 8; do
  RUN_SCRIPT=run_megamoe_colocated_model_suite.sh \
  RESULTS_DIR=/workspace/results/megamoe-colocated-batchfine \
  TOKENS_PER_RANK_GRID="${GRID}" \
  EP_SIZE=${ep} \
  MODELS="qwen3_235b_fp4 minimax_m25_fp4 minimax_m3_fp4" \
  MTP_NEXTN_GRID="0 1 2" \
  BACKEND_GRID=mega \
  WARMUPS=30 ITERATIONS=40 \
  TIME_LIMIT=00:50:00 \
  JOB_NAME="mm-colo-bf2-ep${ep}" \
    ./submit.sh
done

for model in qwen3_235b_fp4 minimax_m25_fp4 minimax_m3_fp4; do
  RUN_SCRIPT=run_megamoe_m2n_model_suite.sh \
  RESULTS_DIR=/workspace/results/megamoe-m2n-batchfine \
  SEQUENCES_PER_AG_RANK_GRID="${GRID}" \
  AFD_SPLIT_GRID="4:4 5:1 6:2 7:1" \
  MICROBATCH_GRID="1 2" \
  MODELS="${model}" \
  MTP_NEXTN_GRID="0 1 2" \
  BACKEND_GRID=mega \
  WARMUPS=30 ITERATIONS=40 \
  TIME_LIMIT=01:30:00 \
  JOB_NAME="mm-bf2-${model%%_*}" \
    ./submit.sh
done
