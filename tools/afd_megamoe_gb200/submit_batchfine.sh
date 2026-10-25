#!/usr/bin/env bash
# Refine the batch grid where the capacity wall actually falls.
#
# The grid steps 64 -> 96 -> 128 -> 192, and on GB200 that gap decides the
# answer. Qwen3-235B at isl=65536, latency limit 50 ms:
#
#   colocated ep8/tp4  best measured rung  b=64   115.3 GiB   16.00 seq/GPU
#                      true capacity wall  b=108  180.9 GiB   27.00
#   AFD 68A4F/tp_a4    best measured rung  b=96   150.1 GiB   22.67
#                      true capacity wall  b=118             27.86
#
# So the sheet reads a 41.7% concurrency advantage for AFD where the hardware
# gives 3.2%. The colocated arm is not out of memory at batch 64 -- it is out of
# RUNGS, sitting at 62% of the card with the next rung 2x away. Refining the
# ladder is the only way to settle that without leaving the measured envelope,
# and sequences_per_ag_rank / logical_tokens_per_rank are free arguments: the
# benchmark only requires > 0 and divisibility by the microbatch count. There is
# no hardware reason 108 was never run; it simply was not in the grid string.
#
# The new rungs bracket the wall from both sides at every context length that
# matters, and all are even so the microbatch=2 suite can use them too.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# 72..120 in steps of 8 fills the 64->128 gap; 160 and 224 fill 128->192->256.
GRID="72 80 88 104 112 120 160 224"

# ---- colocated (fused kernel), ep = 1, 2, 4, 8 -----------------------------
# All four EP values, not just 8: the capacity wall moves with EP because the
# expert weights are sharded differently, and the sweep picks EP per cell.
for ep in 1 2 4 8; do
  RUN_SCRIPT=run_megamoe_colocated_model_suite.sh \
  RESULTS_DIR=/workspace/results/megamoe-colocated-batchfine \
  TOKENS_PER_RANK_GRID="${GRID}" \
  EP_SIZE=${ep} \
  MODELS="qwen3_235b_fp4 minimax_m25_fp4 minimax_m3_fp4" \
  MTP_NEXTN_GRID="0 1 2" \
  BACKEND_GRID=mega \
  WARMUPS=30 ITERATIONS=40 \
  TIME_LIMIT=00:45:00 \
  JOB_NAME="mm-colo-bfine-ep${ep}" \
    ./submit.sh
done

# ---- split (M2N kernel), microbatches 1 and 2 ------------------------------
# The splits that actually win on GB200 all sit at F=4 with a wide A pool, and
# the ag dependence is a small measured correction, so the F value is what has
# to be covered here rather than the exact A. 4:4 and 6:2 and 7:1 give the three
# shards; 5:1 is kept because it is the one the B200 study leaned on.
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
  TIME_LIMIT=01:20:00 \
  JOB_NAME="mm-bfine-${model%%_*}" \
    ./submit.sh
done
