#!/usr/bin/env bash
# Both deliverables for the per-layer skew sweep:
#   1. the projected-layout workbook, built from the per-layer CSV into its own
#      file so the earlier one survives for comparison
#   2. the side-by-side comparison workbook, per-layer vs layer-summed, with the
#      attention-TP question on both readings
#
# The capacity walls are reused from disk on purpose: they are memory limits,
# and routing skew does not move memory. Recomputing them costs a bisection of
# 10-40 AIC calls per cell for an answer that cannot have changed.
# Account and partition come from the environment (SBATCH_ACCOUNT /
# SBATCH_PARTITION) or the command line (sbatch -A ... -p ...).
#SBATCH --job-name=afd-megamoe-books
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
cd "${AFD_TOOLDIR:-$SLURM_SUBMIT_DIR}"
PY=${AIC_PY:?set AIC_PY to the venv python -- see .claude/skills/aic-afd-megamoe/SKILL.md}
: "${AFD_MEGAMOE_ROOT:?set AFD_MEGAMOE_ROOT to the unpacked data bundle}"
export PYTHONPATH=../afd_megamoe_b200:.
echo "node=$(hostname) arch=$(uname -m) started=$(date -Is)"

CSV=handoff/gb200_perlayer_routing.csv
test -s "$CSV" || { echo "missing $CSV -- the sweep did not finish"; exit 1; }
echo "source: $CSV ($(wc -l < "$CSV") lines)"

echo
echo "=== 0/2 capacity walls (memory limits; routing-independent, cached) ==="
if [ -s handoff/capacity_walls.json ]; then
  echo "reusing handoff/capacity_walls.json"
else
  $PY precompute_walls.py --workers 32 --csv "$CSV"
fi

echo
echo "=== 1/2 projected-layout workbook ==="
AFD_SWEEP_CSV=$PWD/$CSV \
AFD_WORKBOOK_OUT=$PWD/handoff/AIC_AFD_MegaMoE_GB200_NVL72_perlayer.xlsx \
  $PY build_gb200_style.py

echo
echo "=== 2/2 comparison workbook ==="
$PY build_skew_comparison.py

echo "finished=$(date -Is)"
